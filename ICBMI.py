#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ICBMI.py -- BALLISTIC MISSILE DEFENCE INTERCEPT DIGITAL TWIN
             (exoatmospheric midcourse hit-to-kill, grounded physics build)
================================================================================

A standalone, single-file, interactive model of the DEFENSIVE half of a
ballistic missile engagement: can a kinetic interceptor, launched from a known
site, be placed on a collision course with an incoming object before that
object reaches its aimpoint -- and how often does that actually work?

WHAT IS MODELLED
    The defence. Sensor detection chain, track quality, predicted intercept
    point, booster energy budget, kill-vehicle divert authority, proportional
    navigation homing, miss distance, and single-shot kill probability.
    The threat appears ONLY as a ballistic arc -- a mass on a Keplerian
    trajectory, derived from ground range alone by the free-flight range
    equation. That is all the defence knows about it and all this file models.

WHAT IS NOT MODELLED, DELIBERATELY
    Threat vehicle design, propulsion, staging, warhead or reentry-vehicle
    physics, and countermeasure / penetration-aid design. None of it is here
    and none of it is needed: an interceptor solves a kinematics problem
    against an object whose arc is set by gravity. The discrimination view
    (mode 7) treats multiple objects as a DEFENCE LIMIT -- how badly the
    interceptor inventory is diluted -- not as a recipe. There is no
    manufacturing guide in this file and one should not be added to it.

NO RAILGUN MODEL -- AND WHY THE ABSENCE IS THE POINT
    This file assesses a railgun-launched interceptor (--railgun, --mkv, and
    the GUN view) and concludes it does not work. It reaches that conclusion
    with equations, margins and measured correlations ONLY. There is no
    geometry, no assembly, no dimensioned drawing and no build description --
    not for the launcher, not for the projectile. The GUN view renders margin
    bars and pass/fail marks and nothing that could be traced.

    A critique and a blueprint are different objects with different reach. A
    critique tells a reader whether a concept closes; a model tells them how
    to make one. Only the first is required to answer the question this
    project asks, and only the second keeps its value in hands the author
    never meant to help. Engineering detail travels once written down -- it
    does not check who is reading it, and it does not stop working when it
    arrives somewhere unwelcome. Convergent designs across supposedly
    independent competitors are the visible evidence of how readily it moves.

    A negative result carries no such risk, because "this fails, and here is
    the number that kills it" is not a manufacturing input. So the salvo
    mathematics is present in full and the hardware is absent. Everything
    needed to EVALUATE the concept is here. Nothing needed to BUILD one is,
    and nothing of that kind should be added later.

THE HONEST HEADLINE
    Run --feasibility. A single exoatmospheric interceptor against a single
    unaccompanied object, with good track, is a solvable kinematics problem and
    the model closes it. Every real-world complication the model then adds --
    track error at handover, divert budget, battlespace timeline, and above all
    the number of objects the defence cannot tell apart -- degrades it faster
    than interceptor performance can recover. The physics of hitting one object
    is not the hard part and never was. This model exists to show where the
    difficulty actually sits.

    Nothing here is validated against a real system. Every parameter is either
    an open published figure or an estimate, and the file marks which.

Dependencies: numpy, pygame

--------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------
    python ICBMI.py                     # interactive viewer (default)
    python ICBMI.py --selftest          # headless physics + render checks
    python ICBMI.py --engage            # one engagement, text report
    python ICBMI.py --feasibility       # honest scorecard
    python ICBMI.py --battlespace       # window and shot count vs range
    python ICBMI.py --discrimination    # how object count dilutes the defence
    python ICBMI.py --architectures     # boost vs midcourse vs terminal
    python ICBMI.py --montecarlo N      # Pk distribution over N engagements
    python ICBMI.py --threat-range KM   # threat ground range (default 10000)
    python ICBMI.py --loft F            # 1.0 = minimum energy, >1 lofted
    python ICBMI.py --objects N         # objects on track (default 1)

--------------------------------------------------------------------------------
CONTROLS
--------------------------------------------------------------------------------
  TAB / 1..9 ........ cycle view mode          SPACE .... play / pause
  mouse drag L ...... orbit globe              wheel .... zoom
  , / . ............. sim speed                R ........ rebuild engagement
  L ................. toggle labels            G ........ toggle lat/lon grid
  UP/DOWN, PGUP/PGDN  scroll (text modes)      H ........ help
  ESC / Q ........... quit
================================================================================
"""

import argparse
import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np


# =============================================================================
# SECTION 1 -- PHYSICAL CONSTANTS AND SYSTEM SPECIFICATION
# =============================================================================

# --- Earth (WGS-84 / EGM reference values) -----------------------------------
MU = 3.986004418e14          # m^3/s^2   standard gravitational parameter
R_E = 6371008.8              # m         mean volumetric radius
R_EQ = 6378137.0             # m         equatorial radius
J2 = 1.08262668e-3           # -         second zonal harmonic
OMEGA_E = 7.2921159e-5       # rad/s     sidereal rotation rate
G0 = 9.80665                 # m/s^2     standard gravity (loss accounting)

# --- Launch platforms ---------------------------------------------------------
# The booster's quoted burnout velocity is ideal. What reaches the intercept is
# that minus gravity and drag losses accrued during the burn, plus whatever the
# platform already had. Modelling this explicitly is what separates the
# platforms -- an air-launched round is not a better rocket, it is the same
# rocket starting above most of the atmosphere with velocity already on it.
#
# `absentee` is the constellation multiplier: how many interceptors must be
# fielded for one to be in position. It is 1 for anything that sits where you
# put it and large for anything in orbit, which is the dominant cost term for
# space basing and the reason it is not simply the best option.
PLATFORMS = {
    "ground_silo": dict(name="Ground silo", alt_m=0.0, v0_ms=0.0,
                        drag_loss_ms=150.0, grav_loss_frac=0.70, absentee=1.0,
                        note="baseline; fixed location, defends what it covers"),
    "sea_based":   dict(name="Sea-based (Aegis-class)", alt_m=0.0, v0_ms=0.0,
                        drag_loss_ms=150.0, grav_loss_frac=0.70, absentee=1.0,
                        note="same energy as silo; buys placement, not reach"),
    "air_launched": dict(name="Air-launched", alt_m=12000.0, v0_ms=250.0,
                        drag_loss_ms=25.0, grav_loss_frac=0.70, absentee=1.0,
                        note="starts above 75% of the atmosphere, already moving"),
    "space_based": dict(name="Space-based (LEO, 1000 km)", alt_m=1000e3,
                        v0_ms=0.0, drag_loss_ms=0.0, grav_loss_frac=0.0,
                        absentee=20.0,
                        note="no losses and already orbital -- but absentee ratio"),
}


def platform_dv(spec, platform="ground_silo"):
    """Burnout velocity actually delivered, after losses and platform head start.

        v_eff = v_burnout + v_platform - drag_loss - g0 * t_burn * grav_frac

    Gravity loss is the dominant term for a surface launch and is why a
    115-second burn gives up roughly 790 m/s before anything else happens.
    """
    p = PLATFORMS[platform]
    grav = G0 * spec["burn_time_s"] * p["grav_loss_frac"]
    return max(100.0, spec["burnout_v_ms"] + p["v0_ms"]
               - p["drag_loss_ms"] - grav)

# --- Vallado exponential atmosphere, Table 8-4 -------------------------------
# (base altitude km, base density kg/m^3, scale height km). Standard reference
# model; far better than a single exponential and adequate for reentry
# deceleration, which is the only place drag matters here.
_ATMOS = np.array([
    [0,    1.225,     7.249], [25,   3.899e-2,  6.349], [30,   1.774e-2,  6.682],
    [40,   3.972e-3,  7.554], [50,   1.057e-3,  8.382], [60,   3.206e-4,  7.714],
    [70,   8.770e-5,  6.549], [80,   1.905e-5,  5.799], [90,   3.396e-6,  5.382],
    [100,  5.297e-7,  5.877], [110,  9.661e-8,  7.263], [120,  2.438e-8,  9.473],
    [130,  8.484e-9, 12.636], [140,  3.845e-9, 16.149], [150,  2.070e-9, 22.523],
    [180,  5.464e-10,29.740], [200,  2.789e-10,37.105], [250,  7.248e-11,45.546],
    [300,  2.418e-11,53.628], [350,  9.518e-12,53.298], [400,  3.725e-12,58.515],
    [450,  1.585e-12,60.828], [500,  6.967e-13,63.822], [600,  1.454e-13,71.835],
    [700,  3.614e-14,88.667], [800,  1.170e-14,124.64], [900,  5.245e-15,181.05],
    [1000, 3.019e-15,268.00],
], dtype=float)

KARMAN_M = 100_000.0         # m   conventional edge of atmosphere
EXO_FLOOR_M = 120_000.0      # m   below this an exo kill vehicle cannot work

# --- Threat model -------------------------------------------------------------
# A threat is characterised by ONE number a defence can actually estimate from
# track: how far it is going. Everything else -- burnout speed, apogee, time of
# flight, reentry speed -- is DERIVED below from the conic range equation.
# Nothing about the vehicle is specified because nothing about it is needed to
# compute where it will be. That is the whole reason midcourse intercept is a
# tractable kinematics problem in the first place.
THREAT_CLASSES = {
    "SRBM":     dict(range_km=  700, label="Short-range        (<1,000 km)"),
    "MRBM":     dict(range_km= 2500, label="Medium-range  (1,000-3,000 km)"),
    "IRBM":     dict(range_km= 4500, label="Intermediate  (3,000-5,500 km)"),
    "ICBM":     dict(range_km=10000, label="Intercontinental   (>5,500 km)"),
    "ICBM_MAX": dict(range_km=13000, label="ICBM, long      (~13,000 km)"),
}

# Generic ballistic coefficient beta = m/(Cd A), kg/m^2. A lumped aerodynamic
# parameter, not a design: it sets only how deep an object penetrates before it
# decelerates, which is what the TERMINAL layer needs to know. Open
# reentry-physics range for a compact body is roughly 5e3 to 1.5e4.
THREAT_BETA = 8000.0

BOOST_TIME_S = 180.0         # s, threat boost phase; open figure, ~3 min

# --- Interceptor specification ------------------------------------------------
# Dimensions are open published figures for a ground-based interceptor of this
# class. Performance values marked "est." are estimates and are the least
# trustworthy numbers in the file -- they drive Pk more than anything else.
INTERCEPTOR = {
    "name": "GBI-class exoatmospheric interceptor",

    # --- booster stack (to scale, millimetres) ---
    "length_mm":         16800.0,   # published overall length ~16.8 m
    "diameter_mm":        1270.0,   # published diameter ~1.27 m
    "stage1_len_mm":      7300.0,
    "stage2_len_mm":      4200.0,
    "stage3_len_mm":      2600.0,
    "shroud_len_mm":      2700.0,

    # --- energy budget ---
    "burnout_v_ms":       6500.0,   # est.; open figures cluster near 6-7 km/s
    "burn_time_s":          115.0,  # est.; three solid stages
    "launch_delay_s":        30.0,  # est.; detection to commit
    "max_loft_deg":          60.0,  # est.

    # --- kill vehicle ---
    "kv_mass_kg":            64.0,  # published EKV-class figure
    "kv_len_mm":           1400.0,
    "kv_dia_mm":            600.0,
    "kv_divert_dv_ms":      200.0,  # est., total divert budget
    "kv_divert_accel_ms2":   60.0,  # est., lateral DACS authority
    "kv_seeker_fov_deg":      2.0,  # est., narrow-FOV LWIR staring seeker
    "kv_seeker_acq_km":     600.0,  # est., acquisition vs compact target
    "kv_seeker_noise_urad":  20.0,  # est., angular noise 1-sigma
    "kv_nav_gain":            4.0,  # textbook proportional-navigation constant
    "kv_guidance_lag_s":      0.2,  # est., first-order autopilot/seeker lag
    "kv_aimpoint_sigma_m":    0.20, # est., boresight + aimpoint bias, 1-sigma
    "kv_lethal_radius_m":     0.5,  # hit-to-kill: body-to-body contact only
    # Kill vehicles carried per booster. 1 is the single-EKV baseline; >1 is
    # the MKV concept -- several vehicles released from one bus, sharing a
    # fire-control solution but each with its own seeker. See --mkv for what
    # that actually buys and the condition under which it stops buying it.
    "kv_count":                 1,
}

# --- Sensor chain -------------------------------------------------------------
# Representative of a large early-warning array and a large tracking radar. All
# detection ranges are COMPUTED from the radar range equation below, never
# asserted -- and they came out badly wrong the first time these were guessed,
# which is the argument for computing them.
#
# `integration_n` is the number of pulses coherently integrated in a dwell, and
# it matters more than intuition suggests: detection range goes as n^(1/4), so
# a long search dwell buys range that peak power cannot. Published EW-radar
# ranges only reconcile with the range equation when the dwell is included.
SENSORS = {
    "ew_radar": {
        "name": "Early-warning radar (UHF phased array)",
        "freq_hz": 440e6, "power_w": 3.0e6, "gain_dbi": 43.0,
        "noise_fig_db": 3.0, "bandwidth_hz": 1e6, "snr_req_db": 12.0,
        "integration_n": 1000, "site_alt_m": 100.0,
    },
    "xband_radar": {
        "name": "Midcourse tracking radar (X-band)",
        "freq_hz": 9.5e9, "power_w": 1.7e6, "gain_dbi": 55.0,
        "noise_fig_db": 3.0, "bandwidth_hz": 1e6, "snr_req_db": 15.0,
        "integration_n": 10000, "site_alt_m": 10.0,
    },
}

# Radar cross-sections, m^2. Generic magnitudes for a compact body; used only
# to compute how far away the DEFENCE can see. UHF sits nearer the resonance
# region for a body of this size, so the figure is larger than at X-band.
RCS_M2 = {"UHF": 0.1, "X": 0.01}

# Track error at handover: 1-sigma cross-range position error, growing with
# coast time since the last sensor update. Estimate. This term dominates miss
# distance, which is the single most important structural fact in the model.
TRACK = {
    "base_sigma_m": 150.0,
    "growth_m_per_s": 4.0,
    "velocity_sigma_ms": 1.5,
    # In-flight target updates. The error above grows only for as long as the
    # track goes UNREFRESHED, so what matters is the gap between the last
    # usable update and handover -- not the whole flight. Modelling growth
    # across the entire engagement assumes the sensor detects once and then
    # looks away, which no real system does and which made the track term
    # roughly 30x more pessimistic than it should be.
    "update_interval_s": 30.0,   # sensor revisit / uplink cadence, est.
    "final_update_lead_s": 30.0, # last usable uplink before handover, est.
}


# =============================================================================
# SECTION 2 -- ENVIRONMENT PHYSICS
# =============================================================================


def atmos_density(alt_m):
    """Piecewise-exponential atmospheric density, kg/m^3 (Vallado Table 8-4)."""
    h_km = alt_m / 1000.0
    if h_km >= 1000.0:
        return 0.0
    if h_km < 0.0:
        h_km = 0.0
    i = int(np.searchsorted(_ATMOS[:, 0], h_km, side="right") - 1)
    i = max(0, min(i, len(_ATMOS) - 1))
    h0, rho0, H = _ATMOS[i]
    return float(rho0 * math.exp(-(h_km - h0) / H))


def gravity(r_vec, use_j2=True):
    """Gravitational acceleration at ECI position r_vec (m), m/s^2, with J2."""
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3)
    a = -MU * r_vec / r**3
    if not use_j2:
        return a
    x, y, z = r_vec
    k = 1.5 * J2 * MU * R_EQ**2 / r**5
    zr2 = 5.0 * (z * z) / (r * r)
    return a + np.array([k * x * (zr2 - 1.0),
                         k * y * (zr2 - 1.0),
                         k * z * (zr2 - 3.0)])


def drag_accel(r_vec, v_vec, beta):
    """Drag acceleration, m/s^2. beta = m/(Cd A) kg/m^2. Uses velocity relative
    to the rotating atmosphere -- at 7 km/s the co-rotation term is small but
    real."""
    alt = float(np.linalg.norm(r_vec)) - R_E
    if alt > 200_000.0 or alt < -1000.0:
        return np.zeros(3)
    rho = atmos_density(alt)
    if rho <= 0.0:
        return np.zeros(3)
    v_atm = np.cross(np.array([0.0, 0.0, OMEGA_E]), r_vec)
    v_rel = v_vec - v_atm
    s = float(np.linalg.norm(v_rel))
    if s < 1e-6:
        return np.zeros(3)
    return -0.5 * rho * s * v_rel / beta


def rk4_step(r, v, dt, beta=None, use_j2=True):
    """One RK4 step of the two-body + J2 (+ optional drag) equations of motion."""
    def acc(rr, vv):
        a = gravity(rr, use_j2)
        if beta is not None:
            a = a + drag_accel(rr, vv, beta)
        return a

    k1v = acc(r, v)
    k1r = v
    k2v = acc(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
    k2r = v + 0.5*dt*k1v
    k3v = acc(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
    k3r = v + 0.5*dt*k2v
    k4v = acc(r + dt*k3r, v + dt*k3v)
    k4r = v + dt*k3v
    r_n = r + (dt/6.0) * (k1r + 2*k2r + 2*k3r + k4r)
    v_n = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return r_n, v_n


# =============================================================================
# SECTION 3 -- BALLISTIC TRAJECTORY
#
# The threat is a mass on a Keplerian arc. Given only ground range, the conic
# range equation fixes burnout speed and flight-path angle. Everything the
# defence needs follows from that and nothing else is required.
# =============================================================================


def conic_range(v_bo, gamma, r_bo, r_imp=R_E):
    """Exact free-flight ground range (m) of a ballistic conic from burnout at
    radius r_bo to impact at radius r_imp.

    Works from the conic elements directly, so it is correct for burnout at
    altitude (asymmetric endpoints) rather than assuming a surface-to-surface
    arc. Returns None if the object escapes or never descends to r_imp.

        Q = v^2 r / mu           e = sqrt(1 + Q(Q-2)cos^2 gamma)
        p = (r v cos gamma)^2 / mu
        r = p / (1 + e cos nu)   ->  nu at each endpoint
    """
    if v_bo <= 0.0:
        return None
    Q = v_bo * v_bo * r_bo / MU
    if Q >= 2.0:
        return None                       # escape trajectory, never returns
    c = math.cos(gamma)
    p = (r_bo * v_bo * c) ** 2 / MU
    e = math.sqrt(max(0.0, 1.0 + Q * (Q - 2.0) * c * c))
    if e < 1e-9:
        return None                       # circular, no impact
    cn_bo = (p / r_bo - 1.0) / e
    cn_im = (p / r_imp - 1.0) / e
    if abs(cn_bo) > 1.0 or abs(cn_im) > 1.0:
        return None                       # never reaches the impact radius
    nu_bo = math.acos(max(-1.0, min(1.0, cn_bo)))       # ascending branch
    nu_im = 2.0 * math.pi - math.acos(max(-1.0, min(1.0, cn_im)))  # descending
    return (nu_im - nu_bo) * R_E


def burnout_for_range(range_km, r_bo, loft=1.0):
    """Burnout speed and flight-path angle for a free-flight arc of the given
    ground range.

    The minimum-energy flight-path angle on a spherical Earth is

        gamma = pi/4 - psi/4,      psi = range / R_E

    which gives 45 degrees at short range (the flat-Earth optimum) and 0 at
    antipodal range (a circular orbit) -- both correct limits, checked in
    selftest. Burnout speed is then bisected against conic_range, which is
    exact and instant, so no propagation is needed to size the arc.

    loft > 1 adds energy and flies a higher, slower arc over the SAME ground
    range; loft < 1 depresses it. Returns (v_ms, gamma_rad).
    """
    target = range_km * 1000.0
    psi = target / R_E
    psi = max(1e-6, min(psi, math.pi * 0.999))
    gamma_me = math.pi / 4.0 - psi / 4.0

    lo, hi = 200.0, 11000.0
    for _ in range(90):
        v = 0.5 * (lo + hi)
        rng = conic_range(v, gamma_me, r_bo)
        if rng is None or rng < target:
            lo = v
        else:
            hi = v
    v_me = 0.5 * (lo + hi)

    if abs(loft - 1.0) < 1e-9:
        return v_me, gamma_me

    # Add energy, then find the flight-path angle that restores the range.
    # Above the minimum-energy angle the range falls as gamma rises (the
    # lofted branch); below it the range falls as gamma drops (depressed).
    v = v_me * (1.0 + 0.08 * abs(loft - 1.0))
    if loft > 1.0:
        glo, ghi = gamma_me, math.radians(86.0)
    else:
        glo, ghi = math.radians(1.0), gamma_me
    for _ in range(90):
        g = 0.5 * (glo + ghi)
        rng = conic_range(v, g, r_bo)
        if rng is None:
            if loft > 1.0:
                ghi = g
            else:
                glo = g
            continue
        if loft > 1.0:
            if rng > target:
                glo = g
            else:
                ghi = g
        else:
            if rng > target:
                ghi = g
            else:
                glo = g
    return v, 0.5 * (glo + ghi)


class Trajectory:
    """A propagated ballistic arc, sampled on a time grid and queryable by
    interpolation."""

    def __init__(self, t, r, v):
        self.t = np.asarray(t)
        self.r = np.asarray(r)
        self.v = np.asarray(v)
        self.alt = np.linalg.norm(self.r, axis=1) - R_E
        self.speed = np.linalg.norm(self.v, axis=1)

    @property
    def tof(self):
        return float(self.t[-1])

    @property
    def apogee_m(self):
        return float(self.alt.max())

    @property
    def apogee_time(self):
        return float(self.t[int(np.argmax(self.alt))])

    @property
    def impact_speed(self):
        return float(self.speed[-1])

    def entry_speed(self):
        """Speed crossing the 100 km entry interface on the way down.

        This is the number usually quoted as 'reentry speed', and it is much
        higher than the speed at the ground: below the entry interface drag
        removes most of the energy. Reporting only one of the two invites
        exactly the confusion this method exists to prevent.
        """
        for i in range(len(self.t) - 1, 0, -1):
            if self.alt[i] >= KARMAN_M:
                return float(self.speed[i])
        return float(self.speed[-1])

    def at(self, t):
        """Cubic Hermite interpolation of (r, v) at time t.

        Linear interpolation of a gravitationally curved arc leaves a sagitta
        error of about (1/8) a dt^2 -- roughly 0.9 m on a 1-second grid at
        intercept altitude, which is twice the kill vehicle's lethal radius.
        It is swamped by track error and never mattered, but position and
        velocity are both already stored, so Hermite is free and removes it.
        """
        t = max(float(self.t[0]), min(float(t), float(self.t[-1])))
        i = int(np.searchsorted(self.t, t) - 1)
        i = max(0, min(i, len(self.t) - 2))
        h = self.t[i+1] - self.t[i]
        if h <= 0:
            return self.r[i].copy(), self.v[i].copy()
        s = (t - self.t[i]) / h
        s2, s3 = s * s, s * s * s
        p0, p1 = self.r[i], self.r[i+1]
        m0, m1 = self.v[i] * h, self.v[i+1] * h
        pos = ((2*s3 - 3*s2 + 1) * p0 + (s3 - 2*s2 + s) * m0
               + (-2*s3 + 3*s2) * p1 + (s3 - s2) * m1)
        vel = ((6*s2 - 6*s) * p0 + (3*s2 - 4*s + 1) * m0
               + (-6*s2 + 6*s) * p1 + (3*s2 - 2*s) * m1) / h
        return pos, vel

    def altitude_at(self, t):
        r, _ = self.at(t)
        return float(np.linalg.norm(r)) - R_E

    def ground_range_m(self):
        """Actual propagated ground range, launch point to impact point."""
        a = self.r[0] / np.linalg.norm(self.r[0])
        b = self.r[-1] / np.linalg.norm(self.r[-1])
        d = float(np.dot(a, b))
        return math.acos(max(-1.0, min(1.0, d))) * R_E


def build_threat(range_km=10000.0, loft=1.0, beta=THREAT_BETA, dt=1.0):
    """Propagate a threat arc of the given free-flight ground range.

    The arc starts at burnout. The boost phase is represented only as a time
    offset, because the defence sees a track, not a vehicle. Boost-phase
    downrange distance (typically a few hundred km) is additional to the
    free-flight range reported here.
    """
    # Burnout altitude for a long-range arc; open figure, ~200 km.
    h_bo = 200_000.0 if range_km > 3000 else 90_000.0
    r_bo = R_E + h_bo
    v_bo, gamma = burnout_for_range(range_km, r_bo, loft)

    # Launch at +X on the equator, flying toward +Y. The whole engagement then
    # lies in the XY plane, which keeps the geometry readable in the viewer.
    r0 = np.array([r_bo, 0.0, 0.0])
    v0 = v_bo * np.array([math.sin(gamma), math.cos(gamma), 0.0])

    ts, rs, vs = [0.0], [r0.copy()], [v0.copy()]
    r, v = r0.copy(), v0.copy()
    t = 0.0
    while t < 6000.0:
        step = dt if (float(np.linalg.norm(r)) - R_E) > 120_000.0 else 0.2
        r, v = rk4_step(r, v, step, beta=beta)
        t += step
        ts.append(t)
        rs.append(r.copy())
        vs.append(v.copy())
        if float(np.linalg.norm(r)) - R_E <= 0.0:
            break

    traj = Trajectory(ts, rs, vs)
    meta = dict(range_km=range_km,
                actual_range_km=traj.ground_range_m() / 1000.0,
                loft=loft, v_burnout_ms=v_bo, gamma_deg=math.degrees(gamma),
                burnout_alt_km=h_bo / 1000.0, boost_time_s=BOOST_TIME_S,
                apogee_km=traj.apogee_m / 1000.0, tof_s=traj.tof,
                entry_speed_ms=traj.entry_speed(),
                impact_speed_ms=traj.impact_speed,
                total_time_s=BOOST_TIME_S + traj.tof)
    return traj, meta


# --- Lambert solver (universal variables, bisection on z) ---------------------


def _c2(z):
    if z > 1e-6:
        return (1.0 - math.cos(math.sqrt(z))) / z
    if z < -1e-6:
        return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)
    return 0.5 - z / 24.0


def _c3(z):
    if z > 1e-6:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s ** 3)
    if z < -1e-6:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s ** 3)
    return 1.0 / 6.0 - z / 120.0


def lambert(r1, r2, tof, prograde=True):
    """Solve Lambert's problem: the conic joining r1 to r2 in time `tof`.

    This is the fire-control solution. Given where the interceptor is and where
    the threat WILL be, it returns the velocity the interceptor must have at
    burnout. Universal-variable formulation, bisected on z for robustness.
    Returns (v1, v2) or None.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))
    if n1 < 1.0 or n2 < 1.0 or tof <= 0.0:
        return None
    cosdnu = max(-1.0, min(1.0, float(np.dot(r1, r2)) / (n1 * n2)))
    dnu = math.acos(cosdnu)
    if (prograde and float(np.cross(r1, r2)[2]) < 0.0) or \
       ((not prograde) and float(np.cross(r1, r2)[2]) > 0.0):
        dnu = 2.0 * math.pi - dnu
    denom = 1.0 - math.cos(dnu)
    if abs(denom) < 1e-12:
        return None
    A = math.sin(dnu) * math.sqrt(n1 * n2 / denom)
    if abs(A) < 1e-9:
        return None

    lo, hi = -4.0 * math.pi ** 2, 4.0 * math.pi ** 2 - 1e-6
    z, y = 0.0, n1 + n2
    for _ in range(80):
        z = 0.5 * (lo + hi)
        C, S = _c2(z), _c3(z)
        if C <= 0.0:
            lo = z
            continue
        y = n1 + n2 + A * (z * S - 1.0) / math.sqrt(C)
        if y < 0.0:
            lo = z
            continue
        x = math.sqrt(y / C)
        t = (x ** 3 * S + A * math.sqrt(y)) / math.sqrt(MU)
        if t < tof:
            lo = z
        else:
            hi = z
        if abs(t - tof) < 1e-3:
            break

    C = _c2(z)
    if C <= 0.0 or y < 0.0:
        return None
    f = 1.0 - y / n1
    g = A * math.sqrt(y / MU)
    gdot = 1.0 - y / n2
    if abs(g) < 1e-9:
        return None
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g


def lambert_best(r1, r2, tof):
    """Cheapest Lambert transfer, trying both transfer directions.

    This matters and is easy to get wrong: the interceptor site sits DOWNRANGE
    of the threat for most of the engagement, so the short way round is often
    the retrograde branch. Solving only prograde silently returns a transfer
    that goes most of the way around the Earth, costs an impossible amount of
    velocity, and makes every intercept look infeasible.
    """
    best = None
    for prograde in (True, False):
        sol = lambert(r1, r2, tof, prograde=prograde)
        if sol is None:
            continue
        dv = float(np.linalg.norm(sol[0]))
        if best is None or dv < best[0]:
            best = (dv, sol)
    return None if best is None else best[1]


# =============================================================================
# SECTION 4 -- SENSORS
#
# Detection range is COMPUTED from the radar range equation, not asserted.
# =============================================================================


def radar_max_range(sensor, rcs_m2):
    """Maximum detection range (m) from the radar range equation:

        R^4 = Pt G^2 lambda^2 sigma n / ( (4pi)^3 k T B F (S/N) )

    with coherent-integration gain n. Every term lives in SENSORS, so the
    number moves when the assumptions do.
    """
    k_b = 1.380649e-23
    T0 = 290.0
    lam = 299_792_458.0 / sensor["freq_hz"]
    G = 10.0 ** (sensor["gain_dbi"] / 10.0)
    F = 10.0 ** (sensor["noise_fig_db"] / 10.0)
    snr = 10.0 ** (sensor["snr_req_db"] / 10.0)
    num = sensor["power_w"] * G * G * lam * lam * rcs_m2 * sensor["integration_n"]
    den = ((4.0 * math.pi) ** 3) * k_b * T0 * sensor["bandwidth_hz"] * F * snr
    return (num / den) ** 0.25


def radar_horizon_m(target_alt_m, site_alt_m=100.0):
    """Geometric horizon range to a target at altitude, m. A radar cannot see
    through the Earth, and for BMD this usually binds well before the range
    equation does -- which is why detection TIME, not radar power, sets how
    much battlespace exists."""
    a = math.sqrt(max(0.0, (R_E + site_alt_m) ** 2 - R_E ** 2))
    b = math.sqrt(max(0.0, (R_E + target_alt_m) ** 2 - R_E ** 2))
    return a + b


def first_detection(traj, sensor, rcs, site_r):
    """Earliest trajectory time at which a sensor at ECI position `site_r` both
    has line of sight and closes the range equation."""
    rmax = radar_max_range(sensor, rcs)
    site_n = float(np.linalg.norm(site_r))
    for i in range(0, len(traj.t), 2):
        rel = traj.r[i] - site_r
        rng = float(np.linalg.norm(rel))
        if rng > rmax:
            continue
        if rng > radar_horizon_m(traj.alt[i], sensor["site_alt_m"]):
            continue
        # target must be above the site's local horizon
        if float(np.dot(rel, site_r)) / (rng * site_n) < -0.02:
            continue
        return float(traj.t[i]), rng, rmax
    return None, None, rmax


# =============================================================================
# SECTION 5 -- INTERCEPTOR AND TERMINAL HOMING
# =============================================================================


def site_position(lat_deg, lon_deg, alt_m=0.0):
    """ECI position of a ground site (epoch-aligned, non-rotating frame)."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    r = R_E + alt_m
    return np.array([r * math.cos(la) * math.cos(lo),
                     r * math.cos(la) * math.sin(lo),
                     r * math.sin(la)])


def platform_state(site_unit, platform="ground_silo", plane_dir=None):
    """Position and pre-existing velocity of the launch platform.

    Every platform starts with velocity it did not have to buy: a surface site
    is carried east by the Earth's rotation, and an orbital platform is already
    moving at orbital speed. Ignoring this understates space basing badly and
    understates a surface launch slightly, so both are carried explicitly.
    """
    p = PLATFORMS[platform]
    r = site_unit * (R_E + p["alt_m"])
    v = np.cross(np.array([0.0, 0.0, OMEGA_E]), r)      # Earth co-rotation
    if platform == "space_based":
        # Circular orbit in the engagement plane, prograde along the threat.
        v_circ = math.sqrt(MU / float(np.linalg.norm(r)))
        d = plane_dir if plane_dir is not None else np.array([0.0, 1.0, 0.0])
        d = d - float(np.dot(d, site_unit)) * site_unit
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            v = d / n * v_circ
    return r, v


def solve_intercept(traj, site_r, t_launch, spec=INTERCEPTOR, t_hi=None,
                    dv_available=None, v_site=None):
    """Find the earliest feasible predicted intercept point (PIP).

    For each candidate intercept time, Lambert returns the velocity the
    interceptor must HAVE at burnout. The velocity it must BUY is the
    difference between that and whatever the platform was already doing --
    Earth co-rotation for a surface site, orbital velocity for a space-based
    one. Charging the full Lambert velocity, as this function did before
    platforms existed, overcharges every shot and understates orbital basing
    severely.

    The shot is feasible when the bought velocity is within the delivered
    energy budget and the intercept is above the exoatmospheric floor.
    Earliest feasible is the useful one: it preserves the most battlespace for
    a second attempt.
    """
    dv_budget = spec["burnout_v_ms"] if dv_available is None else dv_available
    v0 = np.zeros(3) if v_site is None else np.asarray(v_site, float)
    t_ready = t_launch + spec["launch_delay_s"] + spec["burn_time_s"]
    t_hi = traj.tof if t_hi is None else t_hi
    for t_i in np.arange(t_ready + 20.0, t_hi, 5.0):
        r_pip, v_pip = traj.at(t_i)
        alt = float(np.linalg.norm(r_pip)) - R_E
        if alt < EXO_FLOOR_M:
            continue
        tof = float(t_i) - t_ready
        if tof <= 5.0:
            continue
        sol = lambert_best(site_r, r_pip, tof)
        if sol is None:
            continue
        v_req, v_arr = sol
        dv = float(np.linalg.norm(v_req - v0))
        if dv > dv_budget:
            continue
        # `traj` rides along so terminal homing can compute the TRUE
        # differential gravity between the two bodies rather than assuming
        # the relative motion is unaccelerated. At 1,300 km altitude the
        # gravity gradient is ~9e-7 /s^2, which over a 75 s terminal run
        # drifts two objects 4 km apart by about 10 m -- twenty times the
        # kill vehicle's lethal radius, so it is not ignorable.
        return dict(traj=traj, t_intercept=float(t_i), t_launch=t_launch,
                    t_burnout=t_ready, r_pip=r_pip, v_pip=v_pip,
                    v_required=v_req, v_arrival=v_arr, dv_required=dv,
                    dv_budget=dv_budget,
                    closing_speed=float(np.linalg.norm(v_arr - v_pip)),
                    alt_km=alt / 1000.0, flight_time_s=tof,
                    margin_ms=dv_budget - dv)
    return None


def homing_run(sol, spec=INTERCEPTOR, track_sigma_m=None, rng=None, noise=True,
               record=False, shared_track=None):
    """Terminal homing: proportional navigation from seeker acquisition to
    closest approach. Returns (miss_distance_m, divert_used_ms).

    Uses the zero-effort-miss form of PN, which is the correct formulation
    exoatmospherically -- no aerodynamic control exists, so every command is
    spent out of a finite divert budget and cannot be recovered. With
    r_rel = r_interceptor - r_target:

        t_go  = -(r_rel . v_rel) / |v_rel|^2
        ZEM   = r_rel + v_rel t_go            (predicted miss vector)
        a_cmd = -N' ZEM_perp / t_go^2

    The sign is negative because ZEM here points from the target toward where
    the interceptor is predicted to be, so nulling it means accelerating the
    other way. Getting that backwards produces a guidance law that flies the
    interceptor away from the target while looking entirely plausible.

    Two effects keep this from reporting a perfect hit, both physical:
      - a first-order guidance lag, so commands are not achieved instantly;
      - an aimpoint/boresight bias drawn once per engagement. PN nulls to where
        the seeker thinks the target is, so the kill vehicle cannot close
        inside its own aimpoint error no matter how much divert it has.

    Step size is adaptive in t_go and the run terminates on the analytic
    closest approach of the final coast, which is faster and more accurate than
    stepping through it.

    The dominant error is NOT the guidance law. It is where the kill vehicle
    thinks the target is at handover -- `track_sigma_m`.

    `record=True` additionally returns a trace of (elapsed_s, r_rel.copy())
    samples spanning acquisition to closest approach, in the same ECI-aligned
    basis as r_rel itself (built from u/e1/e2, which come straight from the
    ECI v_rel) -- so the trace is directly usable as a 3D flight path, not an
    abstraction that needs reprojecting. This exists so the visualization can
    show the ACTUAL simulated relative motion instead of a stand-in.
    """
    rng = rng or np.random.default_rng()
    trace = [] if record else None
    sigma = TRACK["base_sigma_m"] if track_sigma_m is None else track_sigma_m

    vc = max(sol["closing_speed"], 1.0)
    acq_range = spec["kv_seeker_acq_km"] * 1000.0

    v_rel = sol["v_arrival"] - sol["v_pip"]
    u = v_rel / max(float(np.linalg.norm(v_rel)), 1e-9)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, u))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    # Handover: the kill vehicle is aimed at where the track said the object
    # was, and the seeker takes over from there.
    #
    # `shared_track` lets a caller inject ONE track draw across several kill
    # vehicles released from the same bus. That is the physically correct
    # structure for a salvo: the vehicles share a fire-control solution (so
    # track error is COMMON to all of them) while each carries its own
    # boresight/aimpoint bias and seeker noise (INDEPENDENT). Which of those
    # two dominates decides whether extra vehicles multiply the odds or just
    # buy several copies of the same miss.
    if shared_track is not None:
        off, voff = shared_track
    elif noise:
        off = rng.normal(0.0, sigma, 2)
        voff = rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)
    else:
        off = np.array([sigma, 0.0])
        voff = np.zeros(2)

    # Aimpoint / boresight bias: fixed for the engagement, and irreducible.
    if noise:
        bias = rng.normal(0.0, spec["kv_aimpoint_sigma_m"], 2)
    else:
        bias = np.zeros(2)
    bias_v = e1 * bias[0] + e2 * bias[1]

    r_rel = -u * acq_range + e1 * off[0] + e2 * off[1]
    v_now = v_rel + e1 * voff[0] + e2 * voff[1]

    N = spec["kv_nav_gain"]
    a_max = spec["kv_divert_accel_ms2"]
    budget = spec["kv_divert_dv_ms"]
    tau = max(1e-3, spec["kv_guidance_lag_s"])
    dv_used = 0.0
    elapsed = 0.0
    t_limit = acq_range / vc * 3.0

    if record:
        trace.append((0.0, r_rel.copy()))

    # The integration below runs on scalar components rather than 3-vectors.
    # numpy's per-call overhead dominates at this size -- profiling showed
    # np.linalg.norm alone being entered ~1,200 times per run and accounting
    # for most of the cost. The arithmetic is identical, and the rng calls are
    # untouched and in the same order, so results are unchanged; this is
    # purely the difference between array dispatch and float math.
    # Real differential gravity between the two bodies, when the caller gave
    # us the target's absolute trajectory. Both objects are in free fall, so
    # what perturbs the RELATIVE motion is the difference in the gravity each
    # one feels -- the gravity-gradient term. It is small per second and large
    # over a terminal run, and PN has to spend divert flying it out.
    _traj = sol.get("traj")
    _t_cpa = sol.get("t_intercept")
    _t_hom = acq_range / vc
    use_grav = _traj is not None and _t_cpa is not None
    if use_grav:
        # Target motion across the terminal window, expanded once about the
        # acquisition state instead of interpolating the stored trajectory
        # every step. The gradient term depends on target position only
        # through mu/r^3, so the ~km-level truncation error of a quadratic
        # over 75 s shifts the differential by parts in 10^4 -- far below
        # anything that matters -- while removing a per-step interpolation
        # that cost more than the rest of the loop combined.
        _ra, _va = _traj.at(_t_cpa - _t_hom)
        _ga = gravity(_ra)
        t0x, t0y, t0z = float(_ra[0]), float(_ra[1]), float(_ra[2])
        tvx, tvy, tvz = float(_va[0]), float(_va[1]), float(_va[2])
        tax, tay, taz = float(_ga[0]) * 0.5, float(_ga[1]) * 0.5, float(_ga[2]) * 0.5

    rx, ry, rz = float(r_rel[0]), float(r_rel[1]), float(r_rel[2])
    vx, vy, vz = float(v_now[0]), float(v_now[1]), float(v_now[2])
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    e1x, e1y, e1z = float(e1[0]), float(e1[1]), float(e1[2])
    e2x, e2y, e2z = float(e2[0]), float(e2[1]), float(e2[2])
    bx, by, bz = float(bias_v[0]), float(bias_v[1]), float(bias_v[2])
    ax = ay = az = 0.0
    seeker_sigma = spec["kv_seeker_noise_urad"] * 1e-6

    while elapsed < t_limit:
        vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
        if vmag < 1e-6:
            break
        t_go = -(rx*vx + ry*vy + rz*vz) / (vmag * vmag)
        if t_go <= 0.02:
            break

        # What the seeker reports: true geometry, plus angular noise scaled by
        # range, plus the fixed aimpoint bias it cannot know about.
        mx, my, mz = rx + bx, ry + by, rz + bz
        if noise:
            rmag = math.sqrt(rx*rx + ry*ry + rz*rz)
            ang = rng.normal(0.0, seeker_sigma, 2)
            a0, a1 = float(ang[0]), float(ang[1])
            mx += (e1x*a0 + e2x*a1) * rmag
            my += (e1y*a0 + e2y*a1) * rmag
            mz += (e1z*a0 + e2z*a1) * rmag

        zx, zy, zz = mx + vx*t_go, my + vy*t_go, mz + vz*t_go
        zdotu = zx*ux + zy*uy + zz*uz
        zpx, zpy, zpz = zx - zdotu*ux, zy - zdotu*uy, zz - zdotu*uz
        k_pn = -N / max(t_go * t_go, 1e-6)
        cx, cy, cz = k_pn*zpx, k_pn*zpy, k_pn*zpz
        amag = math.sqrt(cx*cx + cy*cy + cz*cz)
        if amag > a_max:
            sc = a_max / amag
            cx, cy, cz = cx*sc, cy*sc, cz*sc

        dt = max(0.002, min(0.5, t_go / 30.0))

        # First-order lag: the achieved acceleration chases the command.
        lag = min(1.0, dt / tau)
        ax += (cx - ax) * lag
        ay += (cy - ay) * lag
        az += (cz - az) * lag
        step_dv = math.sqrt(ax*ax + ay*ay + az*az) * dt
        if dv_used + step_dv > budget:        # divert exhausted: coast
            ax = ay = az = 0.0
            step_dv = 0.0
        dv_used += step_dv

        # Differential gravity: g(interceptor) - g(target), exact rather than
        # linearised, including J2. Both bodies are falling; only the
        # difference bends the relative path.
        gx = gy = gz = 0.0
        if use_grav:
            tX = t0x + tvx*elapsed + tax*elapsed*elapsed
            tY = t0y + tvy*elapsed + tay*elapsed*elapsed
            tZ = t0z + tvz*elapsed + taz*elapsed*elapsed
            iX, iY, iZ = tX + rx, tY + ry, tZ + rz
            rt = math.sqrt(tX*tX + tY*tY + tZ*tZ)
            ri = math.sqrt(iX*iX + iY*iY + iZ*iZ)
            if rt > 1.0 and ri > 1.0:
                kt = MU / (rt*rt*rt)
                ki = MU / (ri*ri*ri)
                gx = -(ki*iX - kt*tX)
                gy = -(ki*iY - kt*tY)
                gz = -(ki*iZ - kt*tZ)
                # J2 differential, same form, both evaluated exactly.
                ct = 1.5 * J2 * MU * R_EQ*R_EQ / (rt**5)
                ci = 1.5 * J2 * MU * R_EQ*R_EQ / (ri**5)
                zt2 = 5.0 * tZ*tZ / (rt*rt)
                zi2 = 5.0 * iZ*iZ / (ri*ri)
                gx += ci*iX*(zi2 - 1.0) - ct*tX*(zt2 - 1.0)
                gy += ci*iY*(zi2 - 1.0) - ct*tY*(zt2 - 1.0)
                gz += ci*iZ*(zi2 - 3.0) - ct*tZ*(zt2 - 3.0)

        vx += (ax + gx) * dt
        vy += (ay + gy) * dt
        vz += (az + gz) * dt
        rx += vx * dt
        ry += vy * dt
        rz += vz * dt
        elapsed += dt
        if record:
            trace.append((elapsed, np.array([rx, ry, rz])))

    r_rel = np.array([rx, ry, rz])
    v_now = np.array([vx, vy, vz])

    # Analytic closest approach of the final unguided coast.
    vmag = float(np.linalg.norm(v_now))
    if vmag > 1e-9:
        t_go = -float(np.dot(r_rel, v_now)) / (vmag * vmag)
        if t_go > 0.0:
            r_rel = r_rel + v_now * t_go
            elapsed += t_go
    if record:
        trace.append((elapsed, r_rel.copy()))
        return float(np.linalg.norm(r_rel)), dv_used, trace
    return float(np.linalg.norm(r_rel)), dv_used


def single_shot_pk(sol, spec=INTERCEPTOR, track_sigma_m=None, trials=120,
                   seed=0):
    """Single-shot kill probability by Monte Carlo over track and seeker noise.
    Hit-to-kill means contact: miss must fall inside the lethal radius."""
    rng = np.random.default_rng(seed)
    misses, dvs = [], []
    for _ in range(trials):
        m, dv = homing_run(sol, spec, track_sigma_m, rng)
        misses.append(m)
        dvs.append(dv)
    misses = np.array(misses)
    hits = int((misses <= spec["kv_lethal_radius_m"]).sum())
    pk = hits / trials
    lo, hi = wilson_interval(hits, trials)
    return dict(pk=float(pk), pk_lo=lo, pk_hi=hi, pk_halfwidth=(hi - lo) / 2.0,
                trials=trials,
                miss_median=float(np.median(misses)),
                miss_p95=float(np.percentile(misses, 95)),
                dv_median=float(np.median(dvs)),
                dv_p95=float(np.percentile(dvs, 95)),
                misses=misses)


def salvo_kill_probability(sol, spec=INTERCEPTOR, track_sigma_m=None, k=1,
                           trials=400, seed=0):
    """Probability that a salvo of `k` kill vehicles from ONE bus kills the
    target, modelling the correlation structure explicitly.

    Every vehicle in the salvo inherits the SAME fire-control track error --
    one radar, one solution, one release. Each then carries its OWN aimpoint
    bias and seeker noise. So the salvo is neither fully independent (the
    textbook 1-(1-p)^k) nor fully correlated; where it sits between them is
    decided by which error is currently binding:

      * aimpoint-limited (good track)  -> errors are per-vehicle, vehicles
        fail independently, and extra vehicles multiply the odds almost
        exactly as the naive formula predicts.
      * track-limited (poor track)     -> the shared term dominates, vehicles
        miss TOGETHER, and extra vehicles buy far less than the formula says.

    This is the same argument that sinks the unguided-rod salvo in
    report_railgun(); the difference is only WHICH term happens to dominate.
    Reported as `pk` (any vehicle kills), plus `pk_naive` for the comparison
    and `correlation_penalty` for the gap between them.
    """
    sigma = TRACK["base_sigma_m"] if track_sigma_m is None else track_sigma_m
    k = max(1, int(k))
    lethal = spec["kv_lethal_radius_m"]

    # Common random numbers, so results at different k are directly
    # comparable. The track stream depends only on `seed`, and each vehicle's
    # own noise is keyed on (seed, trial, vehicle index) -- so vehicle i in
    # trial t sees identical randomness whether the salvo is 1 vehicle or 6.
    # Without this pairing the k=1 and k=4 runs consume different amounts of
    # randomness, land on different track draws, and can report a LARGER
    # salvo scoring worse than a smaller one -- which is impossible, since
    # the salvo takes the best of k, and is a sampling artefact rather than
    # a result.
    track_rng = np.random.default_rng(seed)
    # One generator per vehicle SLOT, created once. Vehicle i always draws the
    # t-th sample from its own stream, so slot i sees an identical sequence
    # whether the salvo is 1 vehicle or 8 -- exact pairing across k, at the
    # cost of k generators instead of trials*k of them. Building a fresh
    # Generator inside the trial loop is correct but ~50x slower, enough to
    # stall the interactive viewer when the salvo size is raised.
    veh_rngs = [np.random.default_rng([seed, i]) for i in range(k)]

    kills = 0
    vehicle_hits = 0
    best_miss = []
    dv_used = []
    for t_i in range(trials):
        # One track draw shared by the whole salvo.
        off = track_rng.normal(0.0, sigma, 2)
        voff = track_rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)
        hit_any = False
        closest = float("inf")
        for i in range(k):
            m, dv = homing_run(sol, spec, sigma, veh_rngs[i],
                               shared_track=(off, voff))
            closest = min(closest, m)
            dv_used.append(dv)
            if m <= lethal:
                hit_any = True
                vehicle_hits += 1
        best_miss.append(closest)
        if hit_any:
            kills += 1

    pk = kills / trials
    p_single = vehicle_hits / (trials * k)
    lo, hi = wilson_interval(kills, trials)
    return dict(pk=float(pk), pk_lo=lo, pk_hi=hi,
                pk_halfwidth=(hi - lo) / 2.0, trials=trials, k=k,
                p_per_vehicle=float(p_single),
                pk_naive=float(1.0 - (1.0 - p_single) ** k),
                correlation_penalty=float((1.0 - (1.0 - p_single) ** k) - pk),
                miss_median=float(np.median(best_miss)),
                miss_p95=float(np.percentile(best_miss, 95)),
                dv_median=float(np.median(dv_used)),
                dv_p95=float(np.percentile(dv_used, 95)),
                misses=np.array(best_miss))


def salvo_correlation(sol, spec=INTERCEPTOR, track_sigma_m=None, trials=600,
                      seed=0):
    """Measure how correlated two kill vehicles' outcomes actually are.

    Trying to demonstrate independence by comparing P(kill) against
    1-(1-p)^k does not work once p is high: both numbers sit above what a
    few hundred trials can resolve, every row prints the same saturated
    value, and nothing is shown. The correlation between the two vehicles'
    hit/miss outcomes IS measurable at modest sample sizes, and it is the
    actual claim being made.

    Returns the phi coefficient (Pearson correlation on the two binary
    outcomes) plus the joint miss rate against the independent prediction:

        phi ~ 0  -> vehicles fail independently, salvo multiplies the odds
        phi -> 1  -> vehicles fail together, extra vehicles buy little
    """
    sigma = TRACK["base_sigma_m"] if track_sigma_m is None else track_sigma_m
    lethal = spec["kv_lethal_radius_m"]
    track_rng = np.random.default_rng(seed)
    veh_rngs = [np.random.default_rng([seed, i]) for i in range(2)]
    a, b = [], []
    for _t_i in range(trials):
        off = track_rng.normal(0.0, sigma, 2)
        voff = track_rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)
        outs = []
        for i in range(2):
            m, _ = homing_run(sol, spec, sigma, veh_rngs[i],
                              shared_track=(off, voff))
            outs.append(1 if m > lethal else 0)     # 1 == miss
        a.append(outs[0])
        b.append(outs[1])
    a = np.array(a, float)
    b = np.array(b, float)
    pa, pb = a.mean(), b.mean()
    joint = float((a * b).mean())
    denom = math.sqrt(max(1e-12, pa * (1 - pa) * pb * (1 - pb)))
    phi = (joint - pa * pb) / denom if denom > 1e-9 else 0.0
    return dict(phi=float(phi), miss_rate=float((pa + pb) / 2.0),
                joint_miss=joint, joint_if_independent=float(pa * pb),
                trials=trials)


def salvo_outcome_split(sol, spec=INTERCEPTOR, track_sigma_m=None, k=6,
                        trials=300, seed=0):
    """Split salvo outcomes into all-hit / none-hit / mixed.

    This is the diagnostic that explains WHY escalating salvo size stops
    helping. Each trial draws one shared track error, then flies k vehicles
    against it:

      all_hit  -- the shared draw was inside the divert budget; every vehicle
                  succeeds, and the extra ones were not needed
      none_hit -- the shared draw exceeded what divert can fly out; every
                  vehicle fails together, and extra ones cannot help
      mixed    -- the only band where an additional vehicle changes anything

    Adding vehicles buys a share of `mixed` and nothing else. Once `mixed` is
    exhausted the ladder is flat no matter how many rounds are added, because
    `none_hit` is a floor set by an error every vehicle shares.
    """
    sigma = TRACK["base_sigma_m"] if track_sigma_m is None else track_sigma_m
    lethal = spec["kv_lethal_radius_m"]
    k = max(1, int(k))
    track_rng = np.random.default_rng(seed)
    veh_rngs = [np.random.default_rng([seed, i]) for i in range(k)]
    n_all = n_none = 0
    for _ in range(trials):
        off = track_rng.normal(0.0, sigma, 2)
        voff = track_rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)
        hits = [homing_run(sol, spec, sigma, veh_rngs[i],
                           shared_track=(off, voff))[0] <= lethal
                for i in range(k)]
        if all(hits):
            n_all += 1
        elif not any(hits):
            n_none += 1
    return dict(all_hit=n_all / trials, none_hit=n_none / trials,
                mixed=1.0 - (n_all + n_none) / trials, k=k, trials=trials)


def adaptive_salvo_size(sol, spec=INTERCEPTOR, track_sigma_m=None,
                        target_pk=0.99, max_k=10, trials=200, seed=0,
                        stall_gain=0.004, stall_run=3):
    """Implement 'on a miss, add one more vehicle' and report where it lands.

    Escalates salvo size until either the target kill probability is met or
    the ladder stalls -- `stall_run` consecutive additions each worth less
    than `stall_gain`. The stall case is the one that matters: it means the
    remaining failures come from an error the whole salvo shares, so the rule
    would otherwise escalate forever while buying nothing.

    Returns the ladder, the vehicle count that met the target (or None), and
    which of the two ways it terminated.
    """
    ladder = []
    prev = 0.0
    flat = 0
    for k in range(1, max(1, int(max_k)) + 1):
        r = salvo_kill_probability(sol, spec, track_sigma_m, k=k,
                                   trials=trials, seed=seed)
        ladder.append(r["pk"])
        gain = r["pk"] - prev
        if r["pk"] >= target_pk:
            return dict(k=k, pk=r["pk"], ladder=ladder, stalled=False,
                        reason=f"target {target_pk*100:.0f}% met at k={k}")
        flat = flat + 1 if (k > 1 and gain < stall_gain) else 0
        prev = r["pk"]
        if flat >= stall_run:
            return dict(k=None, pk=r["pk"], ladder=ladder, stalled=True,
                        reason=(f"stalled at k={k}: {stall_run} additions "
                                f"worth <{stall_gain*100:.1f} pts each"))
    return dict(k=None, pk=prev, ladder=ladder, stalled=True,
                reason=f"max_k={max_k} reached at {prev*100:.0f}%")


def pct_mc(res):
    """Format a Monte Carlo probability without ever claiming certainty.

    A run where every trial hit does NOT establish P=1; it establishes that P
    is above whatever the sample size can resolve. Printing '100%' there would
    assert something the experiment cannot support -- the same overreach this
    file objects to elsewhere -- so an all-hit result is reported against its
    Wilson lower bound instead, with the trial count that produced it.
    """
    if res["pk"] >= 1.0:
        return f">{_pct(res['pk_lo'])}% (n={res['trials']})"
    return f"{_pct(res['pk'])}%"


def aimpoint_ceiling(sigma_m=None, lethal_m=None):
    """Closed-form upper bound on Pk from aimpoint error alone.

    The kill vehicle nulls to where its seeker believes the target is, so it
    cannot close inside its own boresight/aimpoint bias. With that bias
    Gaussian in two axes, the miss is Rayleigh and

        Pk_max = 1 - exp( -R_lethal^2 / (2 sigma^2) )

    No divert budget, sensor upgrade or guidance change moves this number --
    it is the ceiling everything else runs into. The Monte Carlo reproduces
    it, which is the strongest single validation in this file: an independent
    closed form and a full 3D simulation agreeing to within sampling error.
    """
    s = sigma_m if sigma_m is not None else INTERCEPTOR["kv_aimpoint_sigma_m"]
    R = lethal_m if lethal_m is not None else INTERCEPTOR["kv_lethal_radius_m"]
    if s <= 0:
        return 1.0
    return 1.0 - math.exp(-(R * R) / (2.0 * s * s))


def wilson_interval(hits, trials, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Every Pk in this file is a Monte Carlo estimate, and at the trial counts
    that run interactively the sampling error is a couple of percentage points.
    Reporting the point estimate alone would be the same error this model
    criticises elsewhere -- quoting a probability past the resolution the
    sample supports. Wilson rather than normal-approximation because it stays
    inside [0,1] and behaves at the extremes, where Pk usually sits.
    """
    if trials <= 0:
        return 0.0, 1.0
    p = hits / trials
    d = 1.0 + z * z / trials
    c = p + z * z / (2 * trials)
    s = z * math.sqrt(max(0.0, p * (1 - p) / trials + z * z / (4 * trials * trials)))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


# =============================================================================
# SECTION 6 -- ENGAGEMENT
# =============================================================================


class Engagement:
    """One threat arc, one sensor chain, one interceptor site, end to end."""

    def __init__(self, range_km=10000.0, loft=1.0, objects=1,
                 defence_frac=0.85, spec=INTERCEPTOR, seed=0, trials=120,
                 platform="ground_silo", defence_offset_km=0.0, iftu=True):
        self.spec = spec
        self.objects = max(1, int(objects))
        self.seed = seed
        self.trials = trials
        self.platform = platform
        self.iftu = bool(iftu)
        self.coast_s = None
        self.traj, self.meta = build_threat(range_km, loft)
        self.defence_frac = defence_frac
        self.dv_available = platform_dv(spec, platform)
        self.defence_offset_km = float(defence_offset_km)
        site_unit = self._site_at_time(defence_frac * self.traj.tof) / R_E
        if self.defence_offset_km:
            # Move the site off the threat's ground track. Until this existed
            # every site sat exactly beneath the trajectory, which quietly
            # assumed the interceptor never pays for a plane change -- the most
            # expensive manoeuvre in the problem, and the thing that actually
            # bounds how much area one site defends.
            th = self.defence_offset_km * 1000.0 / R_E
            n = np.array([0.0, 0.0, 1.0])          # threat plane is XY
            site_unit = site_unit * math.cos(th) + n * math.sin(th)
            site_unit = site_unit / np.linalg.norm(site_unit)
        _, v_at = self.traj.at(defence_frac * self.traj.tof)
        self.site_r, self.v_site = platform_state(site_unit, platform, v_at)
        self.placement = None
        self.placement_tried = []
        self.ew_site_r = self._site_at_time(
            min(0.55, defence_frac) * self.traj.tof)
        self.solution = None
        self.pk = None
        self.track_sigma = None
        self.t_last = None
        self.battlespace_s = 0.0
        self.shot_opportunities = 1
        self.shot_ladder = None
        self._run()

    def _place_orbital(self, t_det):
        """Choose where on the engagement plane an orbital platform sits.

        Sweeps station position along the ground track and both orbital
        directions, scoring by the cheapest feasible intercept. Records the
        full spread so the caller can see how much of the platform's value is
        placement luck rather than capability.
        """
        tr = self.traj
        best, tried = None, []
        for frac in np.arange(0.05, 0.98, 0.05):
            unit = self._site_at_time(float(frac) * tr.tof) / R_E
            _, v_at = tr.at(float(frac) * tr.tof)
            for sign, lbl in ((1.0, "prograde"), (-1.0, "retrograde")):
                r_s, v_s = platform_state(unit, "space_based", v_at * sign)
                sol = solve_intercept(tr, r_s, t_det, self.spec,
                                      dv_available=self.dv_available,
                                      v_site=v_s)
                if sol is None:
                    tried.append((frac, lbl, None))
                    continue
                tried.append((frac, lbl, sol["dv_required"]))
                if best is None or sol["dv_required"] < best[0]:
                    best = (sol["dv_required"], r_s, v_s, float(frac), lbl)
        self.placement_tried = tried
        feas = [d for _, _, d in tried if d is not None]
        self.placement = dict(
            n_tried=len(tried), n_feasible=len(feas),
            frac=best[3] if best else None, direction=best[4] if best else None,
            dv=best[0] if best else None)
        if best:
            self.site_r, self.v_site = best[1], best[2]

    def _site_at_time(self, t):
        """A ground site directly beneath the trajectory at time t."""
        r, _ = self.traj.at(t)
        return r / np.linalg.norm(r) * R_E

    def _run(self):
        tr = self.traj
        self.t_ew, self.rng_ew, self.rmax_ew = first_detection(
            tr, SENSORS["ew_radar"], RCS_M2["UHF"], self.ew_site_r)
        self.t_x, self.rng_x, self.rmax_x = first_detection(
            tr, SENSORS["xband_radar"], RCS_M2["X"], self.site_r)

        cands = [t for t in (self.t_ew, self.t_x) if t is not None]
        self.t_detect = min(cands) if cands else None
        if self.t_detect is None:
            return

        if self.platform == "space_based":
            # Must run AFTER detection: the best station depends on when the
            # engagement can start, so optimising against t=0 would place the
            # platform for a shot the defence is not yet allowed to take.
            self._place_orbital(self.t_detect)

        self.solution = solve_intercept(tr, self.site_r, self.t_detect,
                                        self.spec,
                                        dv_available=self.dv_available,
                                        v_site=self.v_site)
        if self.solution is None:
            return

        # Track quality at handover degrades with coast since the last SENSOR
        # UPDATE, not since first detection. With in-flight target updates the
        # relevant gap is the final uplink lead; without them the track is
        # stale for the entire engagement, which is the pessimistic case the
        # model used to assume unconditionally.
        full_coast = self.solution["t_intercept"] - self.t_detect
        self.coast_s = (min(TRACK["final_update_lead_s"], full_coast)
                        if self.iftu else full_coast)
        self.track_sigma = (TRACK["base_sigma_m"]
                            + TRACK["growth_m_per_s"] * self.coast_s)
        # With one kill vehicle these agree exactly (salvo_kill_probability
        # reduces to single_shot_pk at k=1); with more, the salvo model is the
        # correct one because it shares the track draw across the vehicles.
        kvc = int(self.spec.get("kv_count", 1))
        if kvc > 1:
            self.pk = salvo_kill_probability(
                self.solution, self.spec, self.track_sigma, k=kvc,
                trials=self.trials, seed=self.seed)
        else:
            self.pk = single_shot_pk(self.solution, self.spec,
                                     self.track_sigma, trials=self.trials,
                                     seed=self.seed)

        self.t_last = self._last_feasible()
        if self.t_last:
            self.battlespace_s = max(
                0.0, self.t_last - self.solution["t_intercept"])
        self.shot_opportunities = self._shot_count()

        # Build the layered shot ladder: early single/dual shots at 500 km
        # intervals, the main salvo, and trailing single shots after it.
        self.shot_ladder = build_shot_ladder(
            self.traj, self.site_r, self.v_site, self.t_detect,
            self.spec, self.dv_available, spacing_km=500.0,
            dual_shot=True, trials=max(40, self.trials // 2), seed=self.seed)

    def _last_feasible(self):
        """Latest time an intercept is still feasible -- the closing edge of
        the battlespace. Below the exo floor the kill vehicle cannot work."""
        tr = self.traj
        t_ready = (self.t_detect + self.spec["launch_delay_s"]
                   + self.spec["burn_time_s"])
        for t_i in np.arange(tr.tof, self.solution["t_intercept"], -5.0):
            if tr.altitude_at(float(t_i)) < EXO_FLOOR_M:
                continue
            tof = float(t_i) - t_ready
            if tof <= 5.0:
                continue
            sol = lambert_best(self.site_r, tr.at(float(t_i))[0], tof)
            if sol and float(np.linalg.norm(sol[0] - self.v_site)) <= self.dv_available:
                return float(t_i)
        return None

    def _shot_count(self):
        """Sequential shoot-look-shoot attempts that fit in the window. Each
        cycle needs a full boost plus kill assessment."""
        cycle = (self.spec["burn_time_s"] + self.spec["launch_delay_s"] + 30.0)
        if self.battlespace_s <= 0 or cycle <= 0:
            return 1
        return max(1, 1 + int(self.battlespace_s // cycle))

    def leakage(self, interceptors_committed=2):
        """Probability the real object gets through. With `objects`
        indistinguishable candidates the inventory is spread across all of
        them, so only a fraction of the shots go at the real one."""
        if not self.pk:
            return 1.0
        shots = min(interceptors_committed, self.shot_opportunities * self.objects)
        eff = shots / self.objects
        if eff <= 0:
            return 1.0
        return float((1.0 - self.pk["pk"]) ** eff)

    def interceptors_for_leakage(self, target_leak=0.01):
        """How many interceptors hold leakage under a threshold. This is the
        number that actually sizes an inventory."""
        if not self.pk or self.pk["pk"] <= 0.0:
            return None
        n = math.log(target_leak) / math.log(max(1e-9, 1.0 - self.pk["pk"]))
        return int(math.ceil(n * self.objects))



def _ground_range_at_times(traj):
    """Cumulative ground range (m) at each trajectory time index, by
    summing great-circle arcs between consecutive positions. Returns an
    array the same length as traj.t."""
    gr = np.zeros(len(traj.t))
    for i in range(1, len(traj.t)):
        a = traj.r[i - 1] / np.linalg.norm(traj.r[i - 1])
        b = traj.r[i] / np.linalg.norm(traj.r[i])
        d = float(np.dot(a, b))
        gr[i] = gr[i - 1] + math.acos(max(-1.0, min(1.0, d))) * R_E
    return gr


def build_shot_ladder(traj, site_r, v_site, t_detect, spec, dv_available,
                      spacing_km=500.0, dual_shot=True, trials=80, seed=0):
    """Build a layered intercept schedule: early single (or dual) shots at
    regular ground-range intervals along the threat path, then the main
    salvo at the primary intercept point, then trailing single shots if
    the salvo misses.

    The strategy is:
      1. Walk the threat trajectory at `spacing_km` ground-range intervals.
      2. For each candidate point above the exo floor, solve Lambert from
         the interceptor site to that point. If feasible (dv <= budget,
         altitude > 120 km), record it as a shot opportunity.
      3. The main salvo is the earliest feasible shot that uses the full
         kv_count -- typically near the defence_frac point. Early shots
         before it are single or dual. Trailing shots after it are single
         only -- no more salvos after the initial one.

    Each entry in the returned ladder is a dict with:
      - t_intercept, r_pip, v_pip, alt_km, dv_required, closing_speed
      - shot_type: "early", "salvo", or "trailing"
      - kv_count: 1 (early/trailing) or spec["kv_count"] (salvo)
      - pk: result of single_shot_pk or salvo_kill_probability
      - feasible: bool

    The ICBM speed at each point is read from the trajectory, so if the
    threat accelerates or decelerates (e.g. due to drag at lower altitudes),
    each shot's Lambert solution and closing velocity are recalculated
    from the actual state at that time.
    """
    gr = _ground_range_at_times(traj)
    total_range_m = gr[-1]
    spacing_m = spacing_km * 1000.0

    t_ready_base = t_detect + spec["launch_delay_s"] + spec["burn_time_s"]
    kvc = int(spec.get("kv_count", 1))

    # Shoot-look-shoot cycle: time between consecutive probe launches.
    # Each probe fires, we observe the result, then launch the next.
    sls_cycle = spec["burn_time_s"] + spec["launch_delay_s"] + 30.0

    # Find the main salvo time: the earliest feasible intercept from
    # solve_intercept (which already picks the earliest feasible PIP).
    main_sol = solve_intercept(traj, site_r, t_detect, spec,
                               dv_available=dv_available, v_site=v_site)
    salvo_t = main_sol["t_intercept"] if main_sol else None

    # Walk the trajectory at spacing_km intervals
    ladder = []
    salvo_assigned = False

    # --- Phase 1: sequential probe shots ---
    # Each probe fires at a staggered time, targeting the next 500 km
    # ground-range point the threat hasn't reached yet at launch time.
    # This gives each probe a real time-of-flight and a real Lambert
    # solution, rather than all firing at once.
    probe_launch_t = t_ready_base
    probe_range = spacing_m  # first probe targets 500 km downrange
    while salvo_t is not None and probe_launch_t < salvo_t - 30.0:
        # Find the trajectory point at probe_range downrange
        target_found = False
        for i in range(1, len(traj.t)):
            if gr[i] < probe_range:
                continue
            frac = (probe_range - gr[i - 1]) / max(gr[i] - gr[i - 1], 1e-6)
            t_cand = traj.t[i - 1] + frac * (traj.t[i] - traj.t[i - 1])
            r_pip, v_pip = traj.at(t_cand)
            alt = float(np.linalg.norm(r_pip)) - R_E
            break
        else:
            break

        probe_range += spacing_m

        if alt < EXO_FLOOR_M:
            probe_launch_t += sls_cycle
            continue

        tof = float(t_cand) - probe_launch_t
        if tof <= 10.0:
            # Threat reaches this point too soon after launch — skip
            # to the next 500 km point without advancing the clock.
            continue

        # Don't fire probes past the salvo time
        if t_cand > salvo_t:
            break

        sol = lambert_best(site_r, r_pip, tof)

        # Classify: everything before salvo is a probe
        shot_type = "probe"
        kv = 2 if dual_shot else 1

        coast = max(0.0, float(t_cand) - t_detect)
        sigma = TRACK["base_sigma_m"] + TRACK["growth_m_per_s"] * coast

        if sol is None:
            # No Lambert solution — fire directly toward the PIP
            r_dir = r_pip - site_r
            r_dir = r_dir / np.linalg.norm(r_dir)
            v_bo = v_site + dv_available * r_dir
        else:
            v_req, v_arr = sol
            dv = float(np.linalg.norm(v_req - v_site))
            if dv <= dv_available:
                # Feasible! This is actually an early shot, not a probe
                shot_type = "early"
                v_bo = v_req
            else:
                # Infeasible — fire at max capability in Lambert direction
                v_dir = v_req - v_site
                v_dir = v_dir / np.linalg.norm(v_dir)
                v_bo = v_site + dv_available * v_dir

        # Propagate the interceptor on a ballistic arc
        r_int = np.array(site_r, dtype=float)
        v_int = np.array(v_bo, dtype=float)
        dt_probe = max(1.0, tof / 200.0)
        n_steps = int(tof / dt_probe)
        for _ in range(n_steps):
            r_int, v_int = rk4_step(r_int, v_int, dt_probe)
        miss_m = float(np.linalg.norm(r_int - r_pip))
        r_lethal = spec["kv_lethal_radius_m"]
        pk_val = max(0.0, 1.0 - math.exp(
            -r_lethal * r_lethal / (2.0 * miss_m * miss_m)))

        if shot_type == "early":
            # Feasible early shot — use real Pk model
            sol_dict = dict(
                traj=traj, t_intercept=float(t_cand),
                t_launch=float(probe_launch_t),
                t_burnout=float(probe_launch_t),
                r_pip=r_pip, v_pip=v_pip,
                v_required=v_bo, v_arrival=v_int,
                dv_required=float(np.linalg.norm(v_bo - v_site)),
                dv_budget=dv_available,
                closing_speed=float(np.linalg.norm(v_int - v_pip)),
                alt_km=alt / 1000.0, flight_time_s=tof,
                margin_ms=dv_available - float(np.linalg.norm(v_bo - v_site)))
            if kv > 1:
                pk = salvo_kill_probability(sol_dict, spec, sigma, k=kv,
                                            trials=trials, seed=seed)
            else:
                pk = single_shot_pk(sol_dict, spec, sigma,
                                    trials=trials, seed=seed)
        else:
            pk = dict(pk=pk_val, pk_lo=0.0, pk_hi=min(1.0, pk_val * 2),
                      trials=0, miss_median=miss_m,
                      miss_p95=miss_m * 1.5,
                      dv_median=0.0, dv_p95=0.0)

        feasible = (shot_type == "early")
        dv_val = (float(np.linalg.norm(v_bo - v_site))
                  if sol is not None else float('inf'))

        ladder.append(dict(
            t_intercept=float(t_cand), t_launch=float(probe_launch_t),
            r_pip=r_pip, v_pip=v_pip,
            alt_km=alt / 1000.0, dv_required=dv_val,
            closing_speed=float(np.linalg.norm(v_int - v_pip)),
            margin_ms=dv_available - dv_val if sol is not None else float('-inf'),
            shot_type=shot_type, kv_count=kv, pk=pk, feasible=feasible,
            track_sigma=sigma, flight_time_s=tof,
            threat_speed_ms=float(np.linalg.norm(v_pip)),
            r_interceptor=r_int))

        # Advance the probe clock for the next shot
        probe_launch_t += sls_cycle

    # --- Phase 2: salvo + trailing shots (walk remaining 500 km points) ---
    next_range = probe_range  # continue from where probes left off
    for i in range(1, len(traj.t)):
        if gr[i] < next_range:
            continue
        frac = (next_range - gr[i - 1]) / max(gr[i] - gr[i - 1], 1e-6)
        t_cand = traj.t[i - 1] + frac * (traj.t[i] - traj.t[i - 1])
        r_pip, v_pip = traj.at(t_cand)
        alt = float(np.linalg.norm(r_pip)) - R_E

        next_range += spacing_m

        if alt < EXO_FLOOR_M:
            continue

        # Classify: salvo or trailing
        if salvo_t is not None:
            if (not salvo_assigned and abs(t_cand - salvo_t) < 120.0):
                shot_type = "salvo"
                kv = kvc
                salvo_assigned = True
            elif t_cand < salvo_t:
                # Skip — already covered by probes
                continue
            else:
                shot_type = "trailing"
                kv = 1
        else:
            continue

        t_ready = t_ready_base
        tof = float(t_cand) - t_ready
        if tof <= 5.0:
            continue

        sol = lambert_best(site_r, r_pip, tof)
        if sol is None:
            continue
        v_req, v_arr = sol
        dv = float(np.linalg.norm(v_req - v_site))
        feasible = dv <= dv_available
        if not feasible:
            continue

        sol_dict = dict(
            traj=traj, t_intercept=float(t_cand), t_launch=t_detect,
            t_burnout=t_ready, r_pip=r_pip, v_pip=v_pip,
            v_required=v_req, v_arrival=v_arr, dv_required=dv,
            dv_budget=dv_available,
            closing_speed=float(np.linalg.norm(v_arr - v_pip)),
            alt_km=alt / 1000.0, flight_time_s=tof,
            margin_ms=dv_available - dv)

        coast = max(0.0, float(t_cand) - t_detect)
        sigma = TRACK["base_sigma_m"] + TRACK["growth_m_per_s"] * coast

        if kv > 1:
            pk = salvo_kill_probability(sol_dict, spec, sigma, k=kv,
                                        trials=trials, seed=seed)
        else:
            pk = single_shot_pk(sol_dict, spec, sigma, trials=trials, seed=seed)

        ladder.append(dict(
            t_intercept=float(t_cand), t_launch=float(t_ready),
            r_pip=r_pip, v_pip=v_pip,
            alt_km=alt / 1000.0, dv_required=dv,
            closing_speed=sol_dict["closing_speed"],
            margin_ms=dv_available - dv, shot_type=shot_type,
            kv_count=kv, pk=pk, feasible=feasible,
            track_sigma=sigma, flight_time_s=tof,
            threat_speed_ms=float(np.linalg.norm(v_pip))))

    # Ensure the salvo shot is present even if it fell between grid points
    if main_sol is not None:
        salvo_present = any(s["shot_type"] == "salvo" for s in ladder)
        if not salvo_present:
            coast = max(0.0, main_sol["t_intercept"] - t_detect)
            sigma = TRACK["base_sigma_m"] + TRACK["growth_m_per_s"] * coast
            if kvc > 1:
                pk = salvo_kill_probability(main_sol, spec, sigma, k=kvc,
                                            trials=trials, seed=seed)
            else:
                pk = single_shot_pk(main_sol, spec, sigma, trials=trials,
                                    seed=seed)
            ladder.append(dict(
                t_intercept=main_sol["t_intercept"],
                r_pip=main_sol["r_pip"], v_pip=main_sol["v_pip"],
                alt_km=main_sol["alt_km"],
                dv_required=main_sol["dv_required"],
                closing_speed=main_sol["closing_speed"],
                margin_ms=main_sol["margin_ms"], shot_type="salvo",
                kv_count=kvc, pk=pk, feasible=True,
                track_sigma=sigma, flight_time_s=main_sol["flight_time_s"],
                threat_speed_ms=float(np.linalg.norm(main_sol["v_pip"]))))

    ladder.sort(key=lambda s: s["t_intercept"])
    return ladder


def layered_leakage(ladder, objects=1):
    """Probability the threat survives the entire layered shot schedule.

    Each shot is a shoot-look-shoot: if the previous shot killed the
    target, later shots are not needed. The leakage is the product of
    (1 - Pk) across all shots, because every shot must miss for the
    threat to survive. With multiple indistinguishable objects the
    inventory is spread, so each shot's Pk is divided by the object
    count.
    """
    if not ladder:
        return 1.0
    leak = 1.0
    for shot in ladder:
        p = shot["pk"]["pk"] / max(1, objects)
        leak *= (1.0 - min(p, 1.0))
    return float(leak)


# =============================================================================
# SECTION 7 -- ANALYSIS AND REPORTING
# =============================================================================

BAR = "=" * 78
SUB = "-" * 78


def report_engagement(eng):
    m = eng.meta
    print(BAR)
    print(" ENGAGEMENT REPORT -- exoatmospheric midcourse intercept")
    print(BAR)
    print("\n THREAT ARC (derived from ground range alone)")
    print(SUB)
    print(f"   requested free-flight range  {m['range_km']:,.0f} km")
    print(f"   propagated range ........... {m['actual_range_km']:,.0f} km")
    print(f"   loft factor ................ {m['loft']:.2f}"
          f"{'  (minimum energy)' if abs(m['loft']-1) < 1e-6 else '  (off-optimal)'}")
    print(f"   burnout altitude ........... {m['burnout_alt_km']:,.0f} km")
    print(f"   burnout speed .............. {m['v_burnout_ms']/1000:.3f} km/s")
    print(f"   burnout flight-path angle .. {m['gamma_deg']:.1f} deg")
    print(f"   apogee ..................... {m['apogee_km']:,.0f} km")
    print(f"   free-flight time ........... {m['tof_s']/60:.1f} min")
    print(f"   total flight time .......... {m['total_time_s']/60:.1f} min"
          f"   (incl. {m['boost_time_s']:.0f} s boost)")
    print(f"   speed at 100 km entry ...... {m['entry_speed_ms']/1000:.2f} km/s")
    print(f"   speed at ground ............ {m['impact_speed_ms']/1000:.2f} km/s"
          f"   (drag removes the rest)")

    print("\n DETECTION CHAIN (computed from the radar range equation)")
    print(SUB)
    for key, t, rng_, rmax in (("ew_radar", eng.t_ew, eng.rng_ew, eng.rmax_ew),
                               ("xband_radar", eng.t_x, eng.rng_x, eng.rmax_x)):
        print(f"   {SENSORS[key]['name']}")
        print(f"      equation-limited range .. {rmax/1000:,.0f} km")
        if t is None:
            print("      first detection ......... NEVER (horizon or range)")
        else:
            print(f"      first detection ......... T+{t/60:.1f} min "
                  f"at {rng_/1000:,.0f} km")

    if eng.solution is None:
        print("\n   NO FEASIBLE INTERCEPT -- the booster cannot reach any point")
        print("   on the arc above the exoatmospheric floor in the time left.")
        print(BAR)
        return

    s = eng.solution
    print("\n FIRE CONTROL (Lambert solution to the predicted intercept point)")
    print(SUB)
    print(f"   launch ..................... T+{s['t_launch']/60:.1f} min")
    print(f"   burnout .................... T+{s['t_burnout']/60:.1f} min")
    print(f"   intercept .................. T+{s['t_intercept']/60:.1f} min")
    print(f"   intercept altitude ......... {s['alt_km']:,.0f} km")
    print(f"   interceptor flight time .... {s['flight_time_s']:.0f} s")
    print(f"   burnout dv required ........ {s['dv_required']/1000:.3f} km/s")
    print(f"   booster capability ......... {eng.spec['burnout_v_ms']/1000:.3f} km/s")
    print(f"   energy margin .............. {s['margin_ms']/1000:+.3f} km/s")
    print(f"   closing speed .............. {s['closing_speed']/1000:.2f} km/s")

    print("\n TERMINAL HOMING (proportional navigation, ZEM form)")
    print(SUB)
    p = eng.pk
    print(f"   track error at handover .... {eng.track_sigma:,.0f} m (1-sigma)")
    print(f"   seeker acquisition range ... {eng.spec['kv_seeker_acq_km']:,.0f} km")
    print(f"   divert budget .............. {eng.spec['kv_divert_dv_ms']:.0f} m/s")
    print(f"   divert used, median ........ {p['dv_median']:.1f} m/s")
    sat = p['dv_p95'] >= eng.spec['kv_divert_dv_ms'] * 0.99
    print(f"   divert used, p95 ........... {p['dv_p95']:.1f} m/s"
          + ("   <-- BUDGET EXHAUSTED" if sat else ""))
    print(f"   miss distance, median ...... {p['miss_median']:.2f} m")
    print(f"   miss distance, p95 ......... {p['miss_p95']:.2f} m")
    print(f"   lethal radius .............. {eng.spec['kv_lethal_radius_m']:.2f} m")
    print(f"   SINGLE-SHOT Pk ............. {p['pk']*100:.1f}%"
          f"  (95% CI {p['pk_lo']*100:.1f}-{p['pk_hi']*100:.1f}, "
          f"{p['trials']} trials)")
    print("   Pk is a Monte Carlo estimate. Quote the interval, not the point.")

    print("\n BATTLESPACE")
    print(SUB)
    print(f"   window stays open .......... {eng.battlespace_s:,.0f} s")
    print(f"   shoot-look-shoot cycle ..... "
          f"{eng.spec['burn_time_s']+eng.spec['launch_delay_s']+30:.0f} s")
    print(f"   shot opportunities ......... {eng.shot_opportunities}")
    print(f"   objects on track ........... {eng.objects}")
    print(f"   leakage, 2 interceptors .... {eng.leakage(2)*100:.2f}%")
    need = eng.interceptors_for_leakage(0.01)
    print(f"   interceptors for 99% ....... {need if need else 'unreachable'}")
    print(BAR)


def report_layered_shots(range_km=10000.0, objects=1):
    """Report the layered shot schedule: early single/dual shots at 500 km
    intervals, the main salvo, and trailing single shots after it.

    This is the 'fire many times along all spots of the path' strategy:
    single or dual shots probe the threat early and often, each one a
    real Lambert solution with a real Pk. The main salvo fires at the
    primary intercept point. If it misses, trailing single shots continue
    at 500 km intervals -- but no more salvos, only single shots.

    The threat speed at each shot point is read from the trajectory, so
    if the ICBM accelerates or decelerates, each shot is recalculated
    from the actual state at that time.
    """
    eng = Engagement(range_km=range_km, objects=objects, trials=120)
    print(BAR)
    print(" LAYERED SHOT SCHEDULE -- early probes, main salvo, trailing shots")
    print(BAR)

    m = eng.meta
    print(f"\n   threat range ............... {m['range_km']:,.0f} km")
    print(f"   apogee ..................... {m['apogee_km']:,.0f} km")
    print(f"   total flight time .......... {m['total_time_s']/60:.1f} min")
    if eng.t_detect is not None:
        print(f"   first detection ............ T+{eng.t_detect/60:.1f} min")
    else:
        print("   first detection ............ NEVER")
        print(BAR)
        return

    if not eng.shot_ladder:
        print("\n   NO FEASIBLE SHOT LADDER -- no intercept point reachable.")
        print(BAR)
        return

    ladder = eng.shot_ladder
    n_early = sum(1 for s in ladder if s["shot_type"] == "early")
    n_probe = sum(1 for s in ladder if s["shot_type"] == "probe")
    n_salvo = sum(1 for s in ladder if s["shot_type"] == "salvo")
    n_trail = sum(1 for s in ladder if s["shot_type"] == "trailing")
    n_total = len(ladder)
    n_feasible = sum(1 for s in ladder if s.get("feasible", True))
    total_kvs = sum(s["kv_count"] for s in ladder)

    print(f"\n   shot ladder summary")
    print(SUB)
    print(f"   early shots (single/dual) .. {n_early}")
    print(f"   probe shots (fire & hope) .. {n_probe}")
    print(f"   main salvo ................. {n_salvo}")
    print(f"   trailing shots (single) .... {n_trail}")
    print(f"   total shot attempts ........ {n_total}")
    print(f"   of which feasible .......... {n_feasible}")
    print(f"   total kill vehicles used ... {total_kvs}")
    print(f"   spacing .................... 500 km ground range")

    print(f"\n   shot-by-shot detail")
    print(SUB)
    print(f"   {'#':>3} {'type':<8} {'kv':>3} {'feas':>4} {'T+min':>7} {'alt km':>8} "
          f"{'dv km/s':>8} {'close km/s':>10} {'threat km/s':>11} "
          f"{'Pk':>10} {'miss m':>9} {'sigma m':>8}")
    print(f"   {'':>3} {'':<8} {'':>3} {'':>4} {'':>7} {'':>8} "
          f"{'':>8} {'':>10} {'':>11} {'':>10} {'':>9} {'':>8}")

    for i, shot in enumerate(ladder):
        st = shot["shot_type"]
        feas = "yes" if shot.get("feasible", True) else "NO"
        pk_str = pct_mc(shot["pk"])
        dv_str = ("inf" if math.isinf(shot['dv_required'])
                  else f"{shot['dv_required']/1000:.3f}")
        miss_val = shot["pk"].get("miss_median", 0.0)
        miss_str = ("inf" if math.isinf(miss_val)
                    else f"{miss_val:,.0f}")
        print(f"   {i+1:>3} {st:<8} {shot['kv_count']:>3} {feas:>4} "
              f"{shot['t_intercept']/60:>7.1f} "
              f"{shot['alt_km']:>8,.0f} "
              f"{dv_str:>8} "
              f"{shot['closing_speed']/1000:>10.2f} "
              f"{shot['threat_speed_ms']/1000:>11.2f} "
              f"{pk_str:>10} "
              f"{miss_str:>9} "
              f"{shot['track_sigma']:>8,.0f}")

    leak = layered_leakage(ladder, objects)
    print(f"\n   LAYERED LEAKAGE")
    print(SUB)
    print(f"   probability threat survives all {n_total} shots .. "
          f"{leak*100:.4f}%")
    if objects > 1:
        print(f"   (with {objects} indistinguishable objects, Pk per shot "
              f"divided by {objects})")
    print(f"   total kill vehicles committed ...... {total_kvs}")
    print(f"   shots before salvo ................. {n_early + n_probe}")
    print(f"   shots after salvo .................. {n_trail}")
    print(BAR)


def report_feasibility(range_km=10000.0):
    print(BAR)
    print(" FEASIBILITY SCORECARD -- what this model actually closes")
    print(BAR)
    print("""
 The kinematics of hitting one object are solvable and the model closes them.
 What follows is every step where that stops being the whole problem. Each row
 is computed by the runs below, not asserted here.
""")
    base = Engagement(range_km=range_km, objects=1)
    rows = []

    if base.solution and base.pk:
        rows.append(("Reach the arc at all",
                     f"dv {base.solution['dv_required']/1000:.2f} of "
                     f"{INTERCEPTOR['burnout_v_ms']/1000:.2f} km/s available",
                     "CLOSES" if base.solution["margin_ms"] > 0 else "FAILS"))
        rows.append(("Hit it, this track quality",
                     f"Pk {base.pk['pk']*100:.0f}+/-{base.pk['pk_halfwidth']*100:.0f}% at "
                     f"{base.track_sigma:,.0f} m track error",
                     "CLOSES" if base.pk["pk"] > 0.7 else "MARGINAL"))
        # p95 miss, not median: PN nulls the median almost perfectly right up
        # to the point the divert budget saturates, so the median hides the
        # cliff entirely. The tail is where the failure lives.
        for sig in (150.0, 1000.0, 5000.0, 10000.0, 20000.0):
            r = single_shot_pk(base.solution, INTERCEPTOR, sig, trials=100,
                               seed=7)
            rows.append((f"Track error {sig:,.0f} m",
                         f"Pk {r['pk']*100:>3.0f}+/-{r['pk_halfwidth']*100:.0f}%  p95 miss {r['miss_p95']:>8,.0f} m"
                         f"   dv {r['dv_p95']:>3.0f}",
                         "CLOSES" if r["pk"] > 0.7 else
                         ("MARGINAL" if r["pk"] > 0.3 else "FAILS")))
    else:
        rows.append(("Reach the arc at all", "no feasible PIP", "FAILS"))

    for n in (1, 4, 10):
        e = Engagement(range_km=range_km, objects=n, trials=80)
        if e.pk:
            need = e.interceptors_for_leakage(0.01)
            rows.append((f"{n} object(s) on track",
                         f"{need} interceptors for 99%" if need
                         else "unreachable",
                         "CLOSES" if n == 1 else
                         ("MARGINAL" if n <= 4 else "FAILS")))

    w = 28
    print(f" {'STEP'.ljust(w)} {'COMPUTED'.ljust(44)} VERDICT")
    print(" " + SUB)
    for a, b, c in rows:
        print(f" {a.ljust(w)} {b.ljust(44)} {c}")

    print(f"""
 {SUB}
 READ IT THIS WAY

 The first two rows are the part that works, and they are the part that gets
 quoted. A booster with {INTERCEPTOR['burnout_v_ms']/1000:.1f} km/s of burnout velocity really can be placed
 on a collision course with a {range_km:,.0f} km arc, and a kill vehicle with
 {INTERCEPTOR['kv_divert_dv_ms']:.0f} m/s of divert really can null the residual miss. That is solved in
 the same sense orbital rendezvous is solved.

 The track-error block does NOT degrade gracefully, and that is worth reading
 carefully rather than skimming. Kill probability barely moves across the first
 few kilometres of track error, because proportional navigation nulls the miss
 almost perfectly while divert fuel remains. Then the budget saturates and the
 whole thing falls off a cliff over a narrow band. Median miss stays near zero
 on both sides of that cliff, which is why the table reports the 95th
 percentile: the median is the statistic that hides the failure.

 The practical consequence is that divert budget behaves as a threshold, not a
 dial. Below it, more divert buys nothing because miss is already at the
 aimpoint floor; above it, no realistic increase helps, because the fuel needed
 grows with the track error the kill vehicle is trying to fly out. Improving
 the SENSOR moves this cliff; improving the interceptor mostly does not.

 The last block is different in kind. No amount of divert distinguishes one
 object from another, so that block is a counting problem rather than a physics
 problem -- which is exactly why it is the one that does not yield to better
 hardware, and why it is the one that decides whether a defence works.

 This is why the honest description of midcourse defence is that it works
 against what it was told to expect. The model reproduces that shape without
 having been tuned to.
""")
    print(BAR)


def report_architectures(range_km=10000.0):
    print(BAR)
    print(" ARCHITECTURE COMPARISON -- boost / midcourse / terminal")
    print(BAR)
    traj, meta = build_threat(range_km)
    mid_s = meta["tof_s"]
    term = [t for t, a in zip(traj.t, traj.alt)
            if a < KARMAN_M and t > traj.apogee_time]
    term_s = (max(term) - min(term)) if term else 0.0

    print(f"""
 Threat: {range_km:,.0f} km arc, apogee {meta['apogee_km']:,.0f} km, total flight
 {meta['total_time_s']/60:.0f} minutes.

 {SUB}
 PHASE          DURATION    ATTRACTIVE BECAUSE           HARD BECAUSE
 {SUB}
 Boost          {BOOST_TIME_S:>5.0f} s     Object is slow, bright and   Interceptor must already be
                            still whole; one kill ends   within a few hundred km of the
                            the engagement.              launch point, before launch.

 Midcourse      {mid_s/60:>5.1f} min   Longest window by far, and   Vacuum. Nothing separates
                            the only phase with time     objects by drag, so
                            for a second shot.           discrimination is the problem.

 Terminal       {term_s:>5.0f} s     Atmosphere sorts objects     Almost no time, and the
                            by ballistic coefficient --  defended area shrinks to the
                            a free discriminator.        radius reachable in seconds.
 {SUB}

 THE STRUCTURE OF THE PROBLEM

 Each phase fails for a reason the other two do not, and no phase is a superset
 of another. Midcourse has {mid_s/60:.0f} minutes of window and cannot tell what it is
 looking at. Terminal can tell -- atmospheric drag is a discriminator that
 cannot be talked out of sorting light objects from heavy ones -- and has {term_s:.0f}
 seconds to act on it. Boost has both a clean target and a clean signature, and
 requires being somewhere you are usually not allowed to be.

 That is why real programmes are layered rather than optimised. The layers are
 not redundancy; they are three different problems wearing one name. A model
 reporting a single winning architecture would be reporting that it had ignored
 two of the three failure modes.
""")
    print(BAR)


def report_discrimination(range_km=10000.0, max_objects=20):
    print(BAR)
    print(" DISCRIMINATION -- why object count, not interceptor quality, binds")
    print(BAR)
    e1 = Engagement(range_km=range_km, objects=1)
    if not e1.pk:
        print(" No feasible intercept at this range; nothing to dilute.")
        print(BAR)
        return
    p = e1.pk["pk"]
    print(f"""
 Single-shot Pk against one object, at the track quality this engagement
 actually achieves: {p*100:.0f}%.

 In vacuum there is no drag to sort objects by mass, so every object on the
 track stays a candidate until a sensor rules it out. If none can be ruled out,
 interceptors are committed across all of them.

 {SUB}
 OBJECTS   INTERCEPTORS FOR 99%   LEAKAGE WITH 4 COMMITTED
 {SUB}""")
    for n in (1, 2, 4, 6, 10, 15, max_objects):
        need = math.ceil(math.log(0.01) / math.log(max(1e-9, 1.0 - p)) * n)
        leak = (1.0 - p) ** (4.0 / n)
        print(f" {n:>7}   {need:>20}   {leak*100:>22.1f}%")
    print(f"""{SUB}

 The left column grows linearly and the right column collapses. That is the
 whole argument, and it makes no claim about how hard objects are to tell
 apart -- it is what happens when they are NOT told apart.

 Note the direction this is stated in. How many interceptors a defence needs is
 an inventory and budget question, and it is the decision-relevant direction.
 The inverse question is not in this file and should not be added to it.

 It is also why midcourse defence is described as effective against a small,
 unsophisticated or accidental launch and not against a deliberate one. Same
 hardware, different claim, and the entire difference is this table.
""")
    print(BAR)


# --- Gun-launched interceptor assessment --------------------------------------
#
# These are the parameters PROPOSED in INFORNMATIONAL.md, carried here as
# INPUTS TO BE TESTED, not as a design. Nothing in this section optimises,
# improves, or specifies hardware; it only asks whether the proposal as stated
# can do what is claimed, and answers with the physics. The answer is no, and
# the decisive evidence is already inside the source document.
RAILGUN_PROPOSAL = {
    "proj_dia_m":          0.0635,   # 2.5 in, as proposed
    "proj_len_m":          0.0889,   # 3.5 in, as proposed
    "proj_mass_kg":        5.43,     # as proposed
    "barrel_len_m":       10.72,     # as derived in the source document
    "muzzle_v_claimed":  7721.0,     # "Mach 22.5", as claimed later in the doc
    "muzzle_v_earlier":  4322.0,     # "Mach 12.6", as derived earlier in the doc
    "tungsten_rho":     19300.0,     # kg/m^3
    "tungsten_sigma_c":   1.5e9,     # Pa, compressive strength
    "cd_hyper":            0.1,      # optimistic slender-body hypersonic Cd
    "efficiency":          0.35,     # railgun electrical -> kinetic
}


# Candidate projectile materials, for the interactive gun sandbox (mode 10).
# rho kg/m^3; sigma_c compressive strength Pa; t_limit K is the temperature at
# which the material stops being that material in AIR -- melting for metals,
# oxidation/graphitisation for diamond, which is far below its melting point.
MATERIALS = {
    "tungsten":      dict(rho=19300.0, sigma_c=1.5e9,  t_limit=3695.0,
                          note="dense: keeps velocity, but barrel-limited"),
    "CVD diamond":   dict(rho=3515.0,  sigma_c=10.0e9, t_limit=1200.0,
                          note="best in barrel, worst in air: oxidises ~700 C"),
    "steel 4340":    dict(rho=7850.0,  sigma_c=1.5e9,  t_limit=1700.0,
                          note="baseline; no advantage on either axis"),
    "silicon carb.": dict(rho=3210.0,  sigma_c=3.9e9,  t_limit=2000.0,
                          note="light and hard, but brittle in tension"),
    "Ta4HfC5":       dict(rho=14800.0, sigma_c=2.0e9,  t_limit=4215.0,
                          note="highest melting point known (4215 K)"),
}
MATERIAL_NAMES = list(MATERIALS)

STEFAN = 5.670374419e-8


def gun_assessment(v_muzzle, material="tungsten", barrel_len=10.72,
                   nose_r=0.030, proj_len=0.0889, proj_dia=0.0635,
                   efficiency=0.35):
    """Evaluate a gun-launched interceptor at an arbitrary muzzle velocity.

    Four independent physics checks, each returning its own margin so the
    interactive mode can show WHICH one binds as parameters change:

      1. barrel     -- sigma = rho L a must stay under compressive strength
      2. atmosphere -- Allen-Eggers velocity retained climbing out
      3. heating    -- Sutton-Graves flux, radiative-equilibrium wall temp
      4. energy     -- muzzle energy and electrical input per shot

    Caveat on check 3, stated so it is not over-read: the wall temperature is
    the RADIATIVE-EQUILIBRIUM steady state. A projectile crossing the dense
    atmosphere in about a second does not reach it in the bulk -- it ablates
    instead, shedding surface material. So the check should be read as "severe
    ablation" rather than "instantly vaporises". The ordering it produces
    across materials and velocities is sound; the absolute temperature is an
    upper bound.

    Nothing here is a design. It takes a velocity and a material as inputs and
    reports what physics does to them.
    """
    m = MATERIALS[material]
    # Guard the inputs the interactive sandbox can drive to nonsense. A
    # negative velocity makes v^3 negative and the ^0.25 in the radiative
    # balance returns a complex number, which then fails silently in
    # comparisons rather than raising anywhere useful.
    v_muzzle = max(0.0, float(v_muzzle))
    barrel_len = max(0.1, float(barrel_len))
    nose_r = max(1e-3, float(nose_r))
    proj_len = max(1e-3, float(proj_len))
    proj_dia = max(1e-3, float(proj_dia))
    efficiency = min(max(float(efficiency), 1e-3), 1.0)
    A = math.pi * (proj_dia / 2.0) ** 2
    mass = m["rho"] * A * proj_len

    # 1. barrel
    a_req = v_muzzle ** 2 / (2.0 * barrel_len)
    sigma_req = m["rho"] * proj_len * a_req
    a_max = m["sigma_c"] / (m["rho"] * proj_len)
    v_max = math.sqrt(2.0 * a_max * barrel_len)
    barrel_req = v_muzzle ** 2 / (2.0 * a_max)

    # 2. atmosphere (Allen-Eggers, vertical exit)
    beta = mass / (0.1 * A)
    keep = math.exp(-(1.225 * 8400.0) / (2.0 * beta))

    # 3. heating (Sutton-Graves cold wall, then radiative equilibrium)
    q = 1.7415e-4 * math.sqrt(1.225 / max(nose_r, 1e-3)) * v_muzzle ** 3
    t_wall = (q / (0.8 * STEFAN)) ** 0.25

    # 4. energy
    ke = 0.5 * mass * v_muzzle ** 2

    return dict(
        material=material, mass_kg=mass, beta=beta,
        a_req=a_req, sigma_req=sigma_req, sigma_limit=m["sigma_c"],
        stress_ratio=sigma_req / m["sigma_c"],
        v_max_barrel=v_max, barrel_required=barrel_req,
        v_kept=keep, v_after_atmos=v_muzzle * keep,
        q_wm2=q, t_wall=t_wall, t_limit=m["t_limit"],
        thermal_ratio=t_wall / m["t_limit"],
        ke_j=ke, input_j=ke / efficiency,
        pass_barrel=sigma_req <= m["sigma_c"],
        pass_thermal=t_wall <= m["t_limit"],
    )


def railgun_material_limit(p=RAILGUN_PROPOSAL):
    """Muzzle velocity a solid projectile can survive, from material strength.

    While accelerating, the rear face carries the whole projectile mass over
    its own cross-section, so the compressive stress is

        sigma = rho * L * a

    Setting sigma to the material's compressive strength caps acceleration, and
    a barrel of length L_b then caps muzzle velocity at v = sqrt(2 a L_b).
    This is the constraint the source document derives Mach 12.6 from -- and
    then silently drops when it later assumes Mach 22.5 from the same barrel.
    """
    a_max = p["tungsten_sigma_c"] / (p["tungsten_rho"] * p["proj_len_m"])
    v_max = math.sqrt(2.0 * a_max * p["barrel_len_m"])
    a_claim = p["muzzle_v_claimed"] ** 2 / (2.0 * p["barrel_len_m"])
    sigma_claim = p["tungsten_rho"] * p["proj_len_m"] * a_claim
    return dict(a_max=a_max, v_max=v_max, a_claim=a_claim,
                sigma_claim=sigma_claim,
                overstress=sigma_claim / p["tungsten_sigma_c"],
                barrel_required=p["muzzle_v_claimed"] ** 2 / (2.0 * a_max))


def salvo_probability(p_shot, k, common_frac, trials=20000, seed=0):
    """Probability that at least one of k shots hits, when the shots share a
    common aiming error.

    `common_frac` is the fraction of total miss variance that is COMMON to
    every shot in the salvo -- one fire-control solution, one track, one
    atmospheric model, one gun. The textbook formula

        P = 1 - (1 - p)^k

    assumes common_frac = 0, i.e. that every shot fails independently. That is
    the assumption the source document makes, and it is what lets a salvo of 46
    turn an 11% shot into "99.5%".

    It is the wrong assumption for a salvo from a single gun. When the error is
    mostly common, the shots land together: they all hit or all miss, and salvo
    size buys almost nothing. Returns (P_correlated, P_independent).
    """
    rng = np.random.default_rng(seed)
    k = int(k)
    if k < 1:
        return 0.0, 0.0                   # no shots fired, nothing hit
    # Calibrate a miss scale so that a single shot hits with probability p_shot.
    # Rayleigh: P(|N(0,s)_2D| < R) = 1 - exp(-R^2 / 2 s^2), take R = 1.
    p_shot = min(max(p_shot, 1e-6), 1.0 - 1e-6)
    s_tot = math.sqrt(-1.0 / (2.0 * math.log(1.0 - p_shot)))
    s_com = s_tot * math.sqrt(common_frac)
    s_ind = s_tot * math.sqrt(1.0 - common_frac)

    common = rng.normal(0.0, s_com, (trials, 2))[:, None, :]
    indep = rng.normal(0.0, s_ind, (trials, k, 2))
    miss = np.linalg.norm(common + indep, axis=2)
    p_corr = float((miss.min(axis=1) <= 1.0).mean())
    return p_corr, float(1.0 - (1.0 - p_shot) ** k)


def report_railgun():
    """Honest assessment of the gun-launched interceptor proposed in
    INFORNMATIONAL.md. This tests a proposal; it does not design one."""
    p = RAILGUN_PROPOSAL
    lim = railgun_material_limit(p)
    A = math.pi * (p["proj_dia_m"] / 2.0) ** 2
    beta = p["proj_mass_kg"] / (p["cd_hyper"] * A)
    v = p["muzzle_v_claimed"]

    print(BAR)
    print(" GUN-LAUNCHED INTERCEPTOR -- assessment of the INFORNMATIONAL.md proposal")
    print(BAR)
    print(f"""
 This section tests the railgun concept in the source document. It uses that
 document's own numbers as inputs and does not propose, optimise or specify any
 hardware. Four independent checks, any one of which is disqualifying.

 {SUB}
 1. THE BARREL CONTRADICTION  (the decisive one, and it is self-inflicted)
 {SUB}
 A solid projectile under acceleration carries its own mass on its rear face:
     sigma = rho * L * a
 Tungsten compressive strength {p['tungsten_sigma_c']/1e9:.1f} GPa therefore caps acceleration at
 {lim['a_max']:,.0f} m/s^2 ({lim['a_max']/9.81:,.0f} g), and a {p['barrel_len_m']:.2f} m barrel caps muzzle velocity at:

     v_max = sqrt(2 a L) = {lim['v_max']:,.0f} m/s  =  Mach {lim['v_max']/343:.1f}

 That is EXACTLY the Mach 12.6 the document derives -- reproduced here from
 first principles. The document then assumes Mach 22.5 ({v:,.0f} m/s) from the
 same {p['barrel_len_m']:.2f} m barrel, which requires:

     acceleration  {lim['a_claim']:,.0f} m/s^2  ({lim['a_claim']/9.81:,.0f} g)
     stress        {lim['sigma_claim']/1e9:.2f} GPa  vs  {p['tungsten_sigma_c']/1e9:.2f} GPa available
     overstress    {lim['overstress']:.1f}x

 The projectile crushes itself in the barrel. Reaching Mach 22.5 within the
 material limit needs a barrel of {lim['barrel_required']:.1f} m, not {p['barrel_len_m']:.2f} m.

 Every downstream number in the document -- intercept altitude, impact energy,
 the 46-round salvo, the 21,253 MJ -- rests on a muzzle velocity its own
 earlier analysis had already ruled out.

 {SUB}
 2. ATMOSPHERIC EXIT
 {SUB}
 Ballistic coefficient beta = m/(Cd A) = {beta:,.0f} kg/m^2 (high; helps here).
 Allen-Eggers loss climbing out through the atmosphere:""")
    for ang, lbl in ((90.0, "straight up"), (45.0, "45 degrees")):
        loss = math.exp(-(1.225 * 8400.0) / (2.0 * beta * math.sin(math.radians(ang))))
        print(f"     {lbl:12s}  {v:,.0f} -> {v*loss:,.0f} m/s   ({(1-loss)*100:.0f}% lost)")
    q = 1.7415e-4 * math.sqrt(1.225 / 0.03) * v ** 3
    print(f"""
 Survivable on velocity alone. Heating is not: Sutton-Graves stagnation flux at
 sea level and {v:,.0f} m/s is {q/1e6:,.0f} MW/m^2, roughly {q/1e6/5:,.0f}x the peak Apollo
 reentry experienced -- and Apollo met its peak in thin air, not at sea level.
 No coating, SiC or CVD diamond included, holds against that.

 {SUB}
 3. NO TERMINAL GUIDANCE
 {SUB}
 This is the quiet one. A projectile that survived {lim['a_max']/9.81:,.0f} g is a solid slug.
 It has no seeker and no divert, so it is ballistic from the instant it leaves
 the barrel and cannot correct anything.

 Mode 3 of this model shows what that costs. A kill vehicle WITH a {INTERCEPTOR['kv_seeker_acq_km']:.0f} km
 seeker and {INTERCEPTOR['kv_divert_dv_ms']:.0f} m/s of divert still falls to ~35% kill probability once
 track error reaches 10 km, because it runs out of fuel flying out the error.
 Remove the seeker and the divert entirely and the miss is whatever the
 fire-control prediction error was over a 100+ second flight -- kilometres.

 The document assigns this a 5 m, later 50 m, guidance error with no mechanism
 that could produce it. That single unsupported number is doing all the work.

 {SUB}
 4. THE SALVO FALLACY
 {SUB}
 P = 1 - (1-p)^k is only valid when shots fail INDEPENDENTLY. A salvo from one
 gun shares one track, one fire-control solution, one atmospheric model and one
 barrel. Those errors are common: the rounds land together.

 Taking the document's own p = 11% and salvo of 46:
""")
    for cf, lbl in ((0.0, "independent (document's assumption)"),
                    (0.5, "half the error common"),
                    (0.9, "mostly common (one gun, one track)"),
                    (0.99, "almost entirely common")):
        pc, pi = salvo_probability(0.11, 46, cf)
        print(f"     {lbl:36s}  P = {pc*100:5.1f}%"
              + ("   <-- the claimed 99.5%" if cf == 0.0 else ""))
    print(f"""
 The claimed 99.5% exists only in the first row. As soon as the shots share
 their dominant error -- which they must, coming from one gun aimed by one
 solution -- the salvo stops helping and P collapses back toward the
 single-shot value. Buying 46 rounds buys 46 copies of the same miss.

 {SUB}
 VERDICT
 {SUB}
 The concept fails four independent ways, and the first is decided by a number
 the source document itself computed and then set aside. Firing more rounds
 does not fix any of them, because none of the four is a quantity problem.

 What the document's own instinct got right: it kept asking whether the numbers
 were real. That instinct was correct and the numbers were not. Probabilities
 quoted to 0.1% from 30 trials carry roughly +/-8% of standard error -- the
 tables in that file cannot distinguish 27% from 37%, and at one point report
 hit probability RISING with range, which is noise being read as signal.

 The exoatmospheric interceptor modelled in the rest of this file is the
 architecture that does close, and mode 6 shows what it costs.
""")
    print(BAR)


def _pct(x):
    """Format a probability as a percentage without ever rounding to 0 or 100.

    Leakage of 6e-6 is not zero and an intercept probability of 0.999994 is not
    certainty. Printing them as '0.00%' and '100.00%' would reintroduce, purely
    as a formatting artifact, exactly the fabricated certainty the rest of this
    file refuses to produce.
    """
    v = x * 100.0
    if v <= 0.0:
        return "0"
    if v >= 100.0:
        return "100"
    if v < 0.01:
        return f"{v:.1e}"
    if v > 99.99:
        # Choose precision from the size of the complement, so the printed
        # value can never round up to a bare 100.
        comp = 100.0 - v
        digits = max(4, int(math.ceil(-math.log10(comp))) + 2)
        return f"{v:.{digits}f}"
    return f"{v:.2f}"


def report_layered(range_km=10000.0, max_layers=5, objects=1):
    """Independent engagement chains -- the legitimate route to high intercept
    probability, and the only one that survives the correlation argument.

    A salvo from one battery shares one track and one fire-control solution, so
    its errors are common and extra rounds buy little (see report_railgun).
    Separate sites with separate sensors do NOT share those errors, so their
    failures really are independent and probabilities really do compound.

    The catch, and it is the whole point: layering fixes the CORRELATED-ERROR
    problem and does nothing at all for the DISCRIMINATION problem. Object
    count is common to every layer -- every site is looking at the same
    ambiguous cloud. So leakage falls geometrically in layers and only
    linearly in objects, which is why the object axis wins in the end.
    """
    print(BAR)
    print(" LAYERED DEFENCE -- why independence, not volume, raises Pk")
    print(BAR)
    base = Engagement(range_km=range_km, objects=1)
    if not base.pk:
        print(" No feasible intercept at this range.")
        print(BAR)
        return
    p = base.pk["pk"]
    print(f"""
 Single-shot Pk for one engagement chain, at the track quality this engagement
 actually achieves: {p*100:.0f}%.

 Compare two ways of spending {max_layers} interceptors against ONE object:

 {SUB}
 SPENT AS                                        LEAKAGE     INTERCEPT
 {SUB}""")
    for k in range(1, max_layers + 1):
        pc, _ = salvo_probability(p, k, 0.9)         # one battery, shared track
        leak_salvo = 1.0 - pc
        leak_layer = (1.0 - p) ** k                  # k independent chains
        print(f" {k} round salvo, one battery (90% common error)  "
              f"{_pct(leak_salvo):>9}%   {_pct(1-leak_salvo):>9}%")
        print(f" {k} independent site(s), independent tracks     "
              f"{_pct(leak_layer):>9}%   {_pct(1-leak_layer):>9}%")
        if k < max_layers:
            print()
    print(f"""{SUB}

 The two columns start identical at one shot and diverge immediately. That gap
 is the entire practical argument for geographic layering: it is not redundancy
 for its own sake, it is the only way to make the exponent in (1-p)^k real.

 {SUB}
 NOW ADD OBJECTS THE DEFENCE CANNOT SEPARATE
 {SUB}
 LAYERS      1 object     4 objects    10 objects    20 objects
 {SUB}""")
    for k in range(1, max_layers + 1):
        row = f" {k:>4}   "
        for n in (1, 4, 10, 20):
            # Each layer must spread its interceptors across the candidates.
            leak = (1.0 - p) ** (k / n)
            row += f"  {_pct(leak):>9}%"
        print(row)
    print(f"""{SUB}

 Read along a row, then down a column. Down a column leakage falls
 geometrically -- layers compound, because their errors are independent. Along
 a row it decays only as k/n, because every layer is diluted by the same
 ambiguity: the sites are independent, but the object count they face is not.

 This is the honest shape of the whole problem. Independence is purchasable --
 build another site, field another sensor, and the exponent is real. Ambiguity
 is not: no number of independent chains resolves an object each of them
 individually cannot identify, so the n in the denominator never goes away.

 Which is why "near guarantee of a hit" is reachable against ONE object and is
 not reachable by volume of fire against many. Those are different problems,
 and only the first is an engineering problem.
""")
    print(BAR)


def report_mkv(range_km=10000.0, max_kv=6):
    """Multiple kill vehicles per booster: when extra vehicles multiply the
    odds, and the condition under which they stop."""
    print(BAR)
    print(" MULTIPLE KILL VEHICLES -- when more rounds actually help")
    print(BAR)
    e = Engagement(range_km=range_km, trials=30)
    if not e.solution:
        print(" No feasible intercept.")
        print(BAR)
        return
    print(f"""
 Adding rounds is the intuitive fix for a miss, and whether it works depends
 entirely on WHY the miss happened. Vehicles released from one bus share a
 fire-control solution, so track error is common to all of them; each carries
 its own seeker, so aimpoint bias is not. The salvo multiplies the odds only
 to the extent the binding error is the second kind.

 {SUB}
 GOOD TRACK ({e.track_sigma:,.0f} m) -- aimpoint-limited
 {SUB}
 {'KVs':>4}   {'P(kill)':<22}{'naive 1-(1-p)^k':<20}""")
    for k in range(1, max_kv + 1):
        r = salvo_kill_probability(e.solution, e.spec, e.track_sigma, k=k,
                                   trials=260, seed=5)
        print(f" {k:>3}   {pct_mc(r):<22}{_pct(r['pk_naive'])+'%':<20}")
    print(f"""{SUB}
 Past k=2 the measured column stops moving. That is the sample size, not the
 physics: 260 trials cannot resolve a probability of 0.99999, and reporting
 the naive column as if it had been measured would be asserting six decimal
 places that were never observed.

 So the independence claim is tested directly instead, by measuring whether
 two vehicles actually miss together. phi is the correlation between their
 hit/miss outcomes -- 0 means independent, 1 means they fail as one.

 {SUB}
 DO TWO VEHICLES FAIL INDEPENDENTLY?  (phi, measured)
 {SUB}
 {'track sigma':>13}   {'miss rate':<12}{'joint miss':<13}{'if independent':<16}{'phi':>7}""")
    for sg in (270.0, 8000.0, 20000.0, 40000.0):
        c = salvo_correlation(e.solution, e.spec, sg, trials=500, seed=11)
        print(f" {sg:>11,.0f} m   {c['miss_rate']*100:>8.1f}%   "
              f"{c['joint_miss']*100:>9.1f}%   "
              f"{c['joint_if_independent']*100:>12.1f}%   {c['phi']:>+6.2f}")
    print(f"""{SUB}
 At good track phi sits near zero and the joint miss rate matches the
 independent prediction -- the vehicles really do fail separately, because
 what makes them miss is their own boresight error. As track error grows,
 phi climbs toward 1 and the joint miss rate runs far above the independent
 prediction: they are now missing for a reason they share.

 The endpoint is worth stating plainly, because it is stronger than
 "correlation reduces the benefit". Once track error exceeds what the divert
 budget can fly out, every vehicle in the salvo spends its entire fuel load
 correcting toward the same wrong place and arrives there. Sampled at 20 km
 of track error, four vehicles land within about 2 metres of each other,
 roughly 30 km from the target. Extra vehicles then contribute exactly
 nothing -- not less, nothing -- because there is no longer any respect in
 which they differ.

 {SUB}
 WHAT THAT COSTS IN KILL PROBABILITY
 {SUB}
 {'track sigma':>13}   {'1 KV':<20}{'4 KV, measured':<22}{'4 KV if independent':<20}""")
    for sg in (270.0, 3000.0, 8000.0, 20000.0, 40000.0):
        r1 = salvo_kill_probability(e.solution, e.spec, sg, k=1, trials=200,
                                    seed=3)
        r4 = salvo_kill_probability(e.solution, e.spec, sg, k=4, trials=200,
                                    seed=3)
        print(f" {sg:>11,.0f} m   {pct_mc(r1):<20}{pct_mc(r4):<22}"
              f"{_pct(r4['pk_naive'])+'%':<20}")
    print(f"""{SUB}

 WHAT THIS MEANS FOR 'JUST ADD MORE ROUNDS'

 It is a real fix here, and it is worth being precise about why, because the
 same words are wrong elsewhere in this file. Extra kill vehicles work because
 each brings an INDEPENDENT seeker, and the error that is currently binding --
 aimpoint bias -- is a property of the seeker. Two vehicles genuinely get two
 uncorrelated attempts.

 The unguided-rod salvo in --railgun fails the identical test for the opposite
 reason: those rounds carry no seeker at all, so nothing about them is
 independent, and 46 of them inherit one aiming error between them. More
 rounds there buy 46 copies of the same miss.

 So the rule is not 'salvos work' or 'salvos do not work'. It is: a salvo
 multiplies the odds only across whatever the rounds do NOT share. Count the
 independent error sources, not the rounds.

 Note also what MKV does not touch: it raises kill probability against ONE
 object. It does nothing about how many objects are on the track, which is
 the constraint that actually caps the defence (see --discrimination).
""")
    print(BAR)


def report_escalate(range_km=10000.0):
    """Evaluate the rule 'on a miss, add one more round to the salvo'."""
    print(BAR)
    print(" SALVO ESCALATION -- 'on a miss, add one more'")
    print(BAR)
    e = Engagement(range_km=range_km, trials=20)
    if not e.solution:
        print(" No feasible intercept.")
        print(BAR)
        return
    print(f"""
 The rule is intuitive and it is half right. Escalating salvo size is tested
 here directly: fly k vehicles, and if the engagement still fails, add one.

 {SUB}
 WHERE THE LADDER LANDS
 {SUB}
 {'track sigma':>12}   {'P(kill) at k = 1..8':<46}{'outcome'}""")
    for sg in (270.0, 3000.0, 8000.0, 20000.0, 40000.0):
        res = adaptive_salvo_size(e.solution, e.spec, sg, target_pk=0.99,
                                  max_k=8, trials=200, seed=21)
        lad = " ".join(f"{v*100:5.1f}" for v in res["ladder"])
        print(f" {sg:>11,.0f}m   {lad:<46}{res['reason']}")
    print(f"""{SUB}

 At good track the rule terminates immediately -- one extra vehicle is enough
 and a third is already wasted. At degraded track it never terminates: the
 ladder goes flat while still far below target, and the rule would keep
 adding rounds forever.

 {SUB}
 WHY IT STALLS -- outcome split with 6 vehicles flown against one shared track
 {SUB}
 {'track sigma':>12}   {'all 6 hit':>10}{'none hit':>11}{'mixed':>9}   what escalation can reach""")
    for sg in (270.0, 8000.0, 20000.0, 40000.0):
        sp = salvo_outcome_split(e.solution, e.spec, sg, k=6, trials=240,
                                 seed=21)
        print(f" {sg:>11,.0f}m   {sp['all_hit']*100:>9.1f}%"
              f"{sp['none_hit']*100:>10.1f}%{sp['mixed']*100:>8.1f}%"
              f"   only the {sp['mixed']*100:.0f}% mixed band")
    print(f"""{SUB}

 Read the middle column. Those are engagements where every vehicle in the
 salvo missed -- not by chance, but because the shared fire-control error
 exceeded what any of them could fly out with the divert budget available.
 Sampled directly, six vehicles in such a trial land within about a metre of
 each other, thousands of metres from the target. Adding a seventh puts
 another vehicle in the same place.

 So the honest form of the rule is:

   ADD ONE  -- worth it. The second vehicle covers most of the mixed band.
   ADD TWO  -- marginal.
   ADD MORE -- buys nothing measurable, at any track quality.

 And when the ladder stalls below target, the fix is not more rounds. It is
 whatever reduces the SHARED error: a better track, an in-flight update, or
 more divert authority to fly out the error that remains (see --levers and
 --iftu). Escalating the salvo answers a question the failure was not asking.

 The unguided-rod case in --railgun is this same table with the mixed band
 at zero: those rounds carry no seeker, so nothing distinguishes them, and
 the first added round is already wasted.
""")
    print(BAR)


def report_levers(range_km=10000.0):
    """Which knobs actually move kill probability, measured one at a time with
    every other knob set ideal. Answers 'how do I make it more accurate' with
    a ranking rather than a list."""
    print(BAR)
    print(" ACCURACY LEVERS -- what moves Pk, measured with the others ideal")
    print(BAR)
    e = Engagement(range_km=range_km, trials=20)
    if not e.solution:
        print(" No feasible intercept.")
        print(BAR)
        return
    base = dict(INTERCEPTOR)
    print(f"""
 Each lever is swept while the OTHERS are set to ideal, so what shows up is
 the ceiling that lever imposes by itself. Sweeping them jointly hides this:
 a knob looks dead simply because a different one is already binding.

 {SUB}
 AIMPOINT / BORESIGHT ERROR  (sim vs closed form)
 {SUB}
 {'sigma m':>9}{'simulated Pk':>18}{'analytic ceiling':>20}""")
    for ap in (0.05, 0.1, 0.2, 0.4, 0.8):
        sp = dict(base)
        sp["kv_aimpoint_sigma_m"] = ap
        sp["kv_divert_dv_ms"] = 2000.0
        r = single_shot_pk(e.solution, sp, 50.0, trials=400, seed=1)
        print(f" {ap:>8.2f}{r['pk']*100:>13.1f}% +/-{r['pk_halfwidth']*100:>2.0f}"
              f"{aimpoint_ceiling(ap)*100:>19.1f}%")
    print(f"""{SUB}
 The two columns agree to within sampling error. That is a full 3D Monte
 Carlo reproducing an independent closed form, and it means the ceiling is
 real rather than an artefact of the simulation.

 {SUB}
 TRACK ERROR, with aimpoint and divert ideal
 {SUB}""")
    for sg in (500, 5000, 20000, 50000):
        sp = dict(base)
        sp["kv_aimpoint_sigma_m"] = 0.01
        sp["kv_divert_dv_ms"] = 2000.0
        r = single_shot_pk(e.solution, sp, float(sg), trials=300, seed=1)
        print(f" {sg:>8,} m{r['pk']*100:>13.1f}% +/-{r['pk_halfwidth']*100:>2.0f}")
    print(f"""{SUB}
 Track error on its own does almost nothing until it is enormous. It only
 bites through the DIVERT BUDGET -- the kill vehicle spends fuel flying out
 the error, and when the budget runs out the miss goes with it. Track and
 divert are one joint constraint, not two independent ones.

 {SUB}
 THE RANKING
 {SUB}
   1. Aimpoint / boresight error    sets the ceiling. Currently {base['kv_aimpoint_sigma_m']:.2f} m
                                    -> {aimpoint_ceiling()*100:.0f}% and nothing else can beat it.
   2. Divert budget x track error   joint threshold. {base['kv_divert_dv_ms']:.0f} m/s holds until
                                    track error reaches roughly 5 km, then falls off a cliff.
   3. In-flight target updates      insurance, not improvement. Worth ~1 point
                                    at nominal sensor quality; worth 50 points
                                    when the sensor degrades. See --iftu.
   4. Seeker angular noise          no measurable effect (5-80 urad).
   5. Guidance lag                  no measurable effect (0.05-0.6 s).
   6. Divert acceleration           no effect until track error exceeds ~20 km.

 The practical reading: this interceptor is AIMPOINT-LIMITED, not track-limited.
 Buying a better radar, a bigger divert tank or a faster autopilot moves
 nothing at the design point, because none of them is what is binding. Only
 resolving the aimpoint finer -- seeker resolution and target-feature
 selection at the last instant -- raises the ceiling.

 That is worth stating plainly because the intuitive answer is the wrong one.
 'Know where the target is' is what limits the ENGAGEMENT reaching a shot, and
 it is what the discrimination problem is about. It is not what limits the
 shot once taken.
""")
    print(BAR)


def report_iftu(range_km=10000.0):
    """In-flight target updates: insurance against sensor degradation."""
    print(BAR)
    print(" IN-FLIGHT TARGET UPDATES -- worth nothing, until it is worth everything")
    print(BAR)
    print(f"""
 Track error grows with time since the last sensor update. Whether that is the
 whole engagement or just the last uplink interval is the difference between a
 stale track and a refreshed one.

 {SUB}
 SENSOR QUALITY      STALE TRACK              WITH UPDATES
 (error growth)      sigma        Pk          sigma        Pk
 {SUB}""")
    orig = TRACK["growth_m_per_s"]
    try:
        for gw in (2.0, 4.0, 8.0, 12.0, 20.0):
            TRACK["growth_m_per_s"] = gw
            a = Engagement(range_km=range_km, iftu=False, trials=250)
            b = Engagement(range_km=range_km, iftu=True, trials=250)
            if not (a.pk and b.pk):
                continue
            print(f" {gw:>6.0f} m/s      {a.track_sigma:>7,.0f} m {a.pk['pk']*100:>6.1f}%"
                  f"        {b.track_sigma:>6,.0f} m {b.pk['pk']*100:>6.1f}%")
    finally:
        TRACK["growth_m_per_s"] = orig
    print(f"""{SUB}

 At good sensor quality the two columns are the same to within sampling error
 -- the stale track is already inside the divert budget, so refreshing it buys
 nothing measurable. As sensor quality degrades the stale column collapses and
 the refreshed one does not move at all.

 That is the correct way to value in-flight updates: not as an accuracy
 improvement, but as the thing that decouples kill probability from sensor
 quality entirely. A system with updates stops caring how good its radar is,
 within wide limits. A system without them is hostage to it.
""")
    print(BAR)


def footprint_radius_km(range_km=10000.0, platform="ground_silo",
                        spec=INTERCEPTOR, hi=6000.0, iters=9):
    """Largest off-track offset from which a site can still intercept, km.

    Bisects on lateral offset from the threat's ground track. This is the
    quantity that decides how many sites a defence needs, and it cannot be
    read off the interceptor's velocity -- it falls out of where required
    delta-v crosses delivered delta-v.
    """
    lo = 0.0
    if Engagement(range_km=range_km, defence_offset_km=0.0, platform=platform,
                  spec=spec, trials=4).solution is None:
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        e = Engagement(range_km=range_km, defence_offset_km=mid,
                       platform=platform, spec=spec, trials=4)
        if e.solution is not None:
            lo = mid
        else:
            hi = mid
    return lo


def report_footprint():
    """How much area one site defends, and why plane change is nearly free
    from the ground but expensive from orbit."""
    print(BAR)
    print(" DEFENDED FOOTPRINT -- how far off the ground track a site can reach")
    print(BAR)
    print(f"""
 Every engagement elsewhere in this file puts the interceptor site directly
 beneath the threat's ground track. Real sites are offset, and the offset is
 what decides how many of them a defence has to build.

 {SUB}
 THREAT RANGE   FOOTPRINT RADIUS   AREA DEFENDED
 {SUB}""")
    for rk in (2000, 5000, 10000, 13000):
        r = footprint_radius_km(rk)
        area = math.pi * r * r / 1e6
        print(f" {rk:>9,} km   {r:>13,.0f} km   {area:>8.1f} million km2")
    print(f"""{SUB}

 The footprint peaks against mid-range threats. Short-range arcs give the
 interceptor too little time to fly out laterally; long-range arcs make it
 work harder to reach the intercept at all.

 {SUB}
 WHY LATERAL REACH IS NEARLY FREE FROM THE GROUND
 {SUB}
 Moving the site off-track barely changes required delta-v, which looks wrong
 until you notice where the velocity comes from. A booster starting from rest
 buys its velocity in whatever direction it is pointed -- aiming 27 degrees
 out of plane costs essentially nothing extra.

 An orbiting platform cannot do that. It already has {math.sqrt(MU/(R_E+1000e3))/1000:.2f} km/s pointed
 somewhere, and rotating that vector costs 2 v sin(theta/2):
""")
    v_orb = math.sqrt(MU / (R_E + 1000e3))
    for ang in (5, 15, 27, 45):
        print(f"     {ang:>3} degrees of plane change ....... "
              f"{2*v_orb*math.sin(math.radians(ang)/2):>6,.0f} m/s")
    print(f"""
 That asymmetry is why the orbital placement search in --platforms matters so
 much while ground-site offset barely registers. A silo aims; a satellite
 manoeuvres, and manoeuvring is what costs.

 {SUB}
 ONE CAVEAT ON THE MARGIN COLUMN ELSEWHERE
 {SUB}
 solve_intercept() deliberately takes the EARLIEST feasible intercept, because
 that preserves the most battlespace for a second shot. A side effect is that
 the reported energy margin is always near zero -- the earliest feasible shot
 is by definition the most demanding one the booster can just barely make.
 That is a property of the selection rule, not a statement that the design is
 marginal. The footprint edge above is genuinely energy-limited, though: at the
 cliff, no candidate intercept time closes at any margin.
""")
    print(BAR)


def report_platforms(range_km=10000.0):
    """Compare launch platforms on delivered energy and resulting battlespace."""
    print(BAR)
    print(" LAUNCH PLATFORMS -- faster, farther, and what it actually buys")
    print(BAR)
    print(f"""
 Quoted burnout velocity is ideal. What reaches the intercept is that minus
 gravity and drag losses during the burn, plus whatever the platform already
 had. For a {INTERCEPTOR['burn_time_s']:.0f} s burn, gravity alone takes about
 {G0*INTERCEPTOR['burn_time_s']*0.70:,.0f} m/s before anything else happens.

 {SUB}
 PLATFORM                      DELIVERED   Pk   WINDOW  SHOTS   ALT     ABSENTEE
 {SUB}""")
    rows = {}
    for key in PLATFORMS:
        e = Engagement(range_km=range_km, platform=key, trials=90)
        rows[key] = e
        p = PLATFORMS[key]
        if not e.solution or not e.pk:
            print(f" {p['name']:<28}{e.dv_available/1000:>8.2f} km/s"
                  f"   no feasible intercept")
            continue
        print(f" {p['name']:<28}{e.dv_available/1000:>8.2f} km/s"
              f"{e.pk['pk']*100:>5.0f}%{e.battlespace_s:>8.0f}s"
              f"{e.shot_opportunities:>6}{e.solution['alt_km']:>7,.0f}k"
              f"{p['absentee']:>9.0f}x")
    print(SUB)
    for key in PLATFORMS:
        print(f"   {PLATFORMS[key]['name']:<28} {PLATFORMS[key]['note']}")

    sp = rows.get("space_based")
    if sp is not None and sp.placement and sp.placement["n_tried"]:
        pl = sp.placement
        pro = [d for _, l, d in sp.placement_tried if l == "prograde" and d]
        ret = [d for _, l, d in sp.placement_tried if l == "retrograde" and d]
        print(f"""
 {SUB}
 THE ORBITAL CATCH
 {SUB}
 A silo is where you put it. An orbital platform is wherever its orbit has
 carried it, so its position is not a design choice at the moment it is needed.
 Sweeping {pl['n_tried']} stations along the engagement plane:

     usable stations ............ {pl['n_feasible']} of {pl['n_tried']}
     prograde usable ............ {len(pro)}
     retrograde usable .......... {len(ret)}
     best station ............... {pl['frac']*100:.0f}% downrange, {pl['direction']}
     its delta-v ................ {pl['dv']/1000:.2f} km/s

 Stations that fail are not marginal -- parked downrange and prograde, the
 platform must cancel its own orbital velocity to engage something behind it,
 which costs more than launching from the ground. The orbital advantage is
 real and it is conditional on geometry.

 That conditionality is the absentee ratio in physical form: it is why a
 space-based layer is quoted in constellation size rather than interceptor
 count, and why {PLATFORMS['space_based']['absentee']:.0f}x is a plausible multiplier before any
 reliability or reload argument is made.""")

    print(f"""
 {SUB}
 WHAT THIS SAYS ABOUT 'FASTER AND FARTHER'
 {SUB}
 Air launch is the cheapest real gain on this list: same rocket, started above
 most of the atmosphere with velocity already on it, worth roughly
 {platform_dv(INTERCEPTOR,'air_launched')-platform_dv(INTERCEPTOR,'ground_silo'):,.0f} m/s
 of delivered burnout velocity. That is a larger improvement than most changes
 to the interceptor itself, and it is bought entirely by where you start.

 Note what does NOT appear in the Pk column: none of these platforms improves
 accuracy much. They move the intercept earlier and higher and widen the
 window, which buys SHOTS. Accuracy is a sensor property, and no amount of
 launch energy substitutes for knowing where the target is.
""")
    print(BAR)


def report_battlespace():
    print(BAR)
    print(" BATTLESPACE -- window and shot opportunities vs threat range")
    print(BAR)
    print(f"""
 {SUB}
 RANGE      APOGEE    FLIGHT   INTERCEPT   WINDOW   SHOTS     Pk
 (km)       (km)      (min)    alt (km)    (s)
 {SUB}""")
    for rk in (1000, 2500, 4500, 7000, 10000, 13000):
        e = Engagement(range_km=rk, trials=80)
        if not e.solution or not e.pk:
            print(f" {rk:>6,}   {e.meta['apogee_km']:>7,.0f}   "
                  f"{e.meta['total_time_s']/60:>6.1f}      no shot")
            continue
        print(f" {rk:>6,}   {e.meta['apogee_km']:>7,.0f}   "
              f"{e.meta['total_time_s']/60:>6.1f}   {e.solution['alt_km']:>9,.0f}   "
              f"{e.battlespace_s:>6.0f}   {e.shot_opportunities:>5}   "
              f"{e.pk['pk']*100:>5.0f}%")
    print(f"""{SUB}

 Longer-range threats give a midcourse interceptor MORE OPPORTUNITY, which
 reads backwards until you look at the apogee column. A longer arc spends more
 time high and slow, so the window widens and there is room for extra shots.

 But read the Pk column against it, because the two trends point opposite ways.
 Per-shot kill probability drifts DOWN as range grows, because a longer arc
 also means a longer coast between the last sensor update and handover, and
 track error grows with that coast. Longer range buys shot opportunities and
 spends track quality. The net is favourable here only because shots compound
 and track error has not yet reached the divert-saturation cliff.

 The short-range rows are where midcourse defence runs out of geometry: the arc
 never gets high enough or lasts long enough. That regime is exactly what
 terminal-phase systems exist to cover, and it is why no single layer is
 sufficient on its own.
""")
    print(BAR)


# =============================================================================
# SECTION 8 -- INTERACTIVE VISUALISATION (pygame)
# =============================================================================

SCREEN_W, SCREEN_H = 1600, 950

COL_BG     = (8, 11, 18)
COL_PANEL  = (14, 19, 30)
COL_GRID   = (30, 42, 62)
COL_EARTH  = (16, 32, 56)
COL_ATMOS  = (48, 108, 172)
COL_TEXT   = (196, 212, 232)
COL_DIM    = (110, 130, 158)
COL_THREAT = (232, 88, 72)
COL_INTER  = (86, 208, 140)
COL_PIP    = (255, 206, 84)
COL_SENSOR = (128, 168, 255)
COL_ACCENT = (240, 240, 245)
COL_WARN   = (255, 148, 84)
COL_LIMB   = (72, 140, 210)
COL_GLOW   = (40, 80, 130)
COL_HUD    = (22, 30, 46)
COL_EDGE   = (90, 110, 140)
COL_DARK   = (4, 6, 12)

MODES = ("ENGAGEMENT", "INTERCEPTOR", "GEOMETRY", "SENSORS", "TIMELINE",
         "BATTLESPACE", "DISCRIMINATION", "GUN", "LAYERED", "PHYSICS",
         "ABOUT")

# Per-mode key hints. The status bar showed one fixed string listing every
# binding in the program, most of which did nothing in the mode being viewed --
# so the keys that actually mattered were buried. These are the controls that
# do something HERE; global ones are appended separately.
MODE_CONTROLS = {
    "ENGAGEMENT":     "Z terminal close-up   J/K salvo size   A auto-escalate   "
                      "M math panel   drag orbit   wheel zoom",
    "INTERCEPTOR":    "E exploded   X section cut   L labels   drag orbit   "
                      "wheel zoom",
    "GEOMETRY":       "(read-only) miss distribution from the live engagement",
    "SENSORS":        "(read-only) detection ranges computed from the radar "
                      "equation",
    "TIMELINE":       "(read-only) every delay in the engagement, to scale",
    "BATTLESPACE":    "(read-only) first entry computes a sweep; give it a moment",
    "DISCRIMINATION": "(read-only) object count vs a fixed inventory",
    "GUN":            "M material   -/= muzzle velocity   [/] barrel   N nose "
                      "radius",
    "LAYERED":        "[/] independent sites   O/P objects on track",
    "PHYSICS":        "UP/DOWN or wheel to scroll",
    "ABOUT":          "UP/DOWN or wheel to scroll",
}

# One-line captions used by the self-running demo tour (D).
MODE_CAPTIONS = {
    "ENGAGEMENT":     "The whole engagement: threat arc, interceptor arc, and "
                      "the intercept point. Press Z for the collision itself.",
    "INTERCEPTOR":    "The interceptor at true scale -- 16.8 m of booster to "
                      "deliver a 64 kg kill vehicle.",
    "GEOMETRY":       "Closing at ~10 km/s. The miss distribution is what "
                      "decides kill probability.",
    "SENSORS":        "You cannot shoot what you cannot see. Detection time, "
                      "not radar power, sets the battlespace.",
    "TIMELINE":       "Every second between launch detection and intercept is "
                      "time the defence does not get back.",
    "BATTLESPACE":    "Longer-range threats give MORE opportunity -- they fly "
                      "higher and slower.",
    "DISCRIMINATION": "The constraint that actually caps the defence: how many "
                      "objects it cannot tell apart.",
    "GUN":            "The railgun concept, assessed numerically. It fails four "
                      "independent ways. No hardware is drawn.",
    "LAYERED":        "Independence compounds; volume of fire does not. Same "
                      "interceptor count, very different leakage.",
    "PHYSICS":        "Every number in this program comes from these equations, "
                      "evaluated live.",
    "ABOUT":          "Scope, provenance, and what is deliberately left out.",
}

MODE_TITLES = {
    "ENGAGEMENT":     "ENGAGEMENT -- threat arc, interceptor arc, intercept point",
    "INTERCEPTOR":    "INTERCEPTOR -- booster stack and kill vehicle, to scale",
    "GEOMETRY":       "INTERCEPT GEOMETRY -- closing velocity and miss distribution",
    "SENSORS":        "SENSOR CHAIN -- radar range equation and horizon limits",
    "TIMELINE":       "ENGAGEMENT TIMELINE -- where the margin actually is",
    "BATTLESPACE":    "BATTLESPACE -- window and shot count vs threat range",
    "DISCRIMINATION": "DISCRIMINATION -- object count against a fixed inventory",
    "GUN":            "GUN SANDBOX -- drive muzzle velocity and material yourself",
    "LAYERED":        "LAYERED DEFENCE -- independence vs volume of fire",
    "PHYSICS":        "PHYSICS -- the equations, with live values substituted",
    "ABOUT":          "ABOUT -- scope, provenance, and what is deliberately omitted",
}


# =============================================================================
# SECTION 8A -- TRUE-SCALE 3D GEOMETRY (interceptor vehicle)
#
# Built parametrically from INTERCEPTOR's millimetre dimensions, so the solid
# you orbit is the same object the physics uses. Nothing is shrunk to a token
# scale and no dimension is invented for the render.
# =============================================================================


def _ring(r, z, seg):
    return [(r * math.cos(2*math.pi*i/seg), r * math.sin(2*math.pi*i/seg), z)
            for i in range(seg)]


def mesh_frustum(r0, r1, z0, z1, seg=28, cap0=False, cap1=False):
    """Frustum (or cylinder when r0 == r1) along +Z. Returns (verts, faces)."""
    v = _ring(r0, z0, seg) + _ring(r1, z1, seg)
    f = [(i, (i+1) % seg, seg + (i+1) % seg, seg + i) for i in range(seg)]
    if cap0:
        v.append((0.0, 0.0, z0))
        c = len(v) - 1
        f += [(i, c, (i+1) % seg) for i in range(seg)]
    if cap1:
        v.append((0.0, 0.0, z1))
        c = len(v) - 1
        f += [(seg + (i+1) % seg, c, seg + i) for i in range(seg)]
    return np.array(v, dtype=float), f


def build_interceptor_mesh(spec=INTERCEPTOR, seg=28):
    """Assemble the interceptor stack, tail at z=0, nose at +Z, in millimetres.

    Stage lengths come straight from the spec dict, so the rendered stack is
    exactly `length_mm` long and any change to the spec moves the geometry.
    """
    R = spec["diameter_mm"] / 2.0
    z = 0.0
    parts = []

    def add(name, verts, faces, col, group):
        parts.append(dict(name=name, verts=verts, faces=faces, color=col,
                          group=group, zc=float(verts[:, 2].mean())))

    # Stage 1 nozzle bell (expands aft, below z=0)
    v, f = mesh_frustum(R * 0.34, R * 0.72, -900.0, 0.0, seg)
    add("Stage 1 nozzle", v, f, (78, 74, 70), 0)
    # Nozzle throat (narrow constriction at the top of the bell)
    v, f = mesh_frustum(R * 0.28, R * 0.34, -1050.0, -900.0, seg)
    add("", v, f, (62, 58, 54), 0)

    # Interstages are carved OUT of each stage's allocation rather than added
    # to it, so the assembled stack is exactly spec["length_mm"] long. Adding
    # them on top would silently render a vehicle longer than the spec.
    IS12, IS23 = 300.0, 260.0

    s1 = spec["stage1_len_mm"]
    v, f = mesh_frustum(R, R, z, z + s1 - IS12, seg, cap0=True)
    add(f"Stage 1 solid motor  ({s1/1000:.1f} m)", v, f, (52, 68, 94), 0)
    # Aft fins on stage 1 (4 trapezoidal fins)
    for i in range(4):
        a = 2 * math.pi * i / 4
        fin_w, fin_h = R * 0.35, 1200.0
        fv = np.array([
            [R, 0, 0],
            [R + fin_w, 0, fin_h * 0.2],
            [R + fin_w, 0, fin_h * 0.8],
            [R, 0, fin_h],
        ], dtype=float)
        ff = [(0, 1, 2, 3)]
        rot = np.array([[math.cos(a), 0, math.sin(a)],
                        [0, 1, 0],
                        [-math.sin(a), 0, math.cos(a)]])
        fv = fv @ rot.T
        add("", fv, ff, (44, 58, 82), 0)
    z += s1 - IS12

    v, f = mesh_frustum(R, R * 0.97, z, z + IS12, seg)
    add("Interstage 1/2", v, f, (40, 52, 72), 1)
    z += IS12

    s2 = spec["stage2_len_mm"]
    v, f = mesh_frustum(R * 0.97, R * 0.97, z, z + s2 - IS23, seg)
    add(f"Stage 2 solid motor  ({s2/1000:.1f} m)", v, f, (62, 82, 112), 1)
    z += s2 - IS23

    v, f = mesh_frustum(R * 0.97, R * 0.88, z, z + IS23, seg)
    add("Interstage 2/3", v, f, (40, 52, 72), 2)
    z += IS23

    s3 = spec["stage3_len_mm"]
    v, f = mesh_frustum(R * 0.88, R * 0.88, z, z + s3, seg)
    add(f"Stage 3 solid motor  ({s3/1000:.1f} m)", v, f, (74, 98, 132), 2)
    z += s3

    # Payload shroud, then the kill vehicle carried inside it
    sh = spec["shroud_len_mm"]
    shroud_z0 = z
    v, f = mesh_frustum(R * 0.88, R * 0.62, z, z + sh * 0.62, seg)
    add("Payload shroud", v, f, (104, 132, 168), 3)
    v, f = mesh_frustum(R * 0.62, R * 0.10, z + sh * 0.62, z + sh, seg,
                        cap1=True)
    add("Nose fairing", v, f, (126, 156, 192), 3)

    kz = shroud_z0 + 260.0
    for p in build_kv_mesh(spec, seg):
        add(p["name"], p["verts"] + np.array([0.0, 0.0, kz]), p["faces"],
            p["color"], 4)

    return parts


def build_kv_mesh(spec=INTERCEPTOR, seg=28):
    """Kill vehicle alone, base at local origin, nose toward +Z, millimetres.

    By the time terminal homing starts the booster stages have separated --
    the KV is the only thing still flying and the only thing that can be
    shown colliding with (or missing) the target. Factored out here, in its
    own coordinate frame, so it can be placed and oriented independently of
    the full stack instead of dragging along ~16 m of spent booster whenever
    it is rendered on its own. `build_interceptor_mesh` calls this and
    translates it into the assembled stack, so the two views can never drift
    out of agreement with each other.
    """
    kv_l, kv_r = spec["kv_len_mm"], spec["kv_dia_mm"] / 2.0
    parts = []

    def add(name, verts, faces, col):
        parts.append(dict(name=name, verts=verts, faces=faces, color=col,
                          group=0, zc=float(verts[:, 2].mean())))

    v, f = mesh_frustum(kv_r, kv_r, 0.0, kv_l, seg, cap0=True)
    add(f"Exoatmospheric kill vehicle  ({spec['kv_mass_kg']:.0f} kg)",
        v, f, (206, 168, 74))
    v, f = mesh_frustum(kv_r * 0.55, kv_r * 0.34, kv_l, kv_l + 240.0, seg,
                        cap1=True)
    add("LWIR seeker aperture", v, f, (232, 226, 210))

    for i in range(4):
        a = 2 * math.pi * i / 4
        vv, ff = mesh_frustum(70.0, 96.0, 0.0, 150.0, 10, cap1=True)
        rot = np.array([[math.cos(a), 0, math.sin(a)],
                        [0, 1, 0],
                        [-math.sin(a), 0, math.cos(a)]])
        vv = vv @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float).T
        vv = vv + np.array([0.0, kv_r, 0.0])
        vv = vv @ rot.T + np.array([0.0, 0.0, kv_l * 0.42])
        add("Divert thruster (DACS)" if i == 0 else "", vv, ff, (236, 132, 64))

    return parts


def build_threat_rv_mesh(seg=20):
    """Simple conical reentry vehicle mesh, nose at +Z, tail at z=0, in mm.

    Represents a generic ICBM reentry vehicle: a blunt cone with a small
    base cylinder. Dimensions are approximate for a 300 kg RV.
    """
    base_r = 300.0   # mm base radius
    nose_r = 30.0    # mm nose radius (blunt)
    body_l = 1800.0  # mm body length
    flare_l = 300.0  # mm flare section
    parts = []

    def add(name, verts, faces, col, group):
        parts.append(dict(name=name, verts=verts, faces=faces, color=col,
                          group=group, zc=float(verts[:, 2].mean())))

    # Main conical body (nose cone)
    v, f = mesh_frustum(base_r, nose_r, 0.0, body_l, seg, cap0=True, cap1=True)
    add("Reentry vehicle", v, f, (180, 60, 50), 0)

    # Aft flare (widens slightly at the base)
    v, f = mesh_frustum(base_r, base_r * 1.15, -flare_l, 0.0, seg, cap0=True)
    add("Aft flare", v, f, (140, 44, 36), 0)

    # Fins (4 small trapezoidal fins at the base)
    for i in range(4):
        a = 2 * math.pi * i / 4
        fin_w, fin_h = base_r * 0.25, 400.0
        fv = np.array([
            [base_r * 1.15, 0, -flare_l],
            [base_r * 1.15 + fin_w, 0, -flare_l * 0.7],
            [base_r * 1.15 + fin_w, 0, -flare_l * 0.3],
            [base_r * 1.15, 0, 0],
        ], dtype=float)
        ff = [(0, 1, 2, 3)]
        rot = np.array([[math.cos(a), 0, math.sin(a)],
                        [0, 1, 0],
                        [-math.sin(a), 0, math.cos(a)]])
        fv = fv @ rot.T
        add("", fv, ff, (120, 36, 30), 0)

    return parts


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Camera:
    """Orbit camera. Scene units are ECI metres."""

    def __init__(self):
        self.az = math.radians(-58.0)
        self.el = math.radians(16.0)
        self.dist = 34e6
        self.center = np.array([R_E * 0.5, R_E * 0.5, 0.0])
        self.pan = [0.0, 0.0]

    def matrix(self):
        return rot_x(self.el) @ rot_y(self.az)

    def project(self, p, rect):
        cx = rect.x + rect.w / 2.0 + self.pan[0]
        cy = rect.y + rect.h / 2.0 + self.pan[1]
        focal = min(rect.w, rect.h) * 1.05
        q = (np.asarray(p, float) - self.center) @ self.matrix().T
        z = q[2] + self.dist
        if z <= 1e3:
            return None
        return (cx + focal * q[0] / z, cy - focal * q[1] / z, z)

    def cam_dir(self):
        """Unit vector from the scene toward the camera, in world space."""
        return self.matrix().T @ np.array([0.0, 0.0, -1.0])


class App:
    def __init__(self, args):
        import pygame
        import pygame.gfxdraw
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(
            "ICBMI -- Ballistic Missile Defence Intercept Digital Twin")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H),
                                              pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.font_s = pygame.font.SysFont("consolas", 13)
        self.font_b = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_t = pygame.font.SysFont("consolas", 26, bold=True)

        self.args = args
        self.mode = 0
        self.cam = Camera()
        self.playing = True
        self.sim_t = 0.0
        self.sim_speed = 12.0
        self.labels = True
        self.grid = True
        self.scroll = 0
        self.show_help = False
        self.dragging = False
        self.last_mouse = None
        self._bs_cache = None
        # Gun sandbox state (mode 10) -- start at the material-limited Mach 12.6
        # the source document itself derived, so the first thing on screen is
        # the honest number rather than the aspirational one.
        # Vehicle camera (mode 2): scene units are millimetres, so it needs its
        # own centre and range -- the globe camera works in ECI metres.
        self.cam_v = Camera()
        self.cam_v.center = np.array([0.0, 0.0, INTERCEPTOR["length_mm"] / 2.0])
        self.cam_v.dist = 46000.0
        self.cam_v.az = math.radians(-32.0)
        self.cam_v.el = math.radians(14.0)
        self._mesh = None
        self._kv_mesh = None
        self._rv_mesh = None
        self.explode = False
        self.zoom_intercept = False  # snap-zoom to PIP for collision view
        self.cam_z = Camera()        # dedicated camera for the collision view
        self.zoom_scale = 1.0        # user-applied multiplier on auto-framing
        self.auto_escalate = False   # on a missed terminal run, add a vehicle
        self.show_math = True        # live intercept-math panel in zoom view
        self.demo = False            # self-running guided tour
        self.demo_t = 0.0
        self.demo_hold = 11.0        # seconds per mode in the tour
        self.escalate_log = []       # what escalation actually bought
        self.probe_paths = []        # animated probe shot trajectories
        self.section = False
        self.gun_v = 4322.0
        self.gun_mat = 0
        self.gun_barrel = 10.72
        self.gun_nose = 0.030
        # Layered defence state (mode 11)
        self.lay_layers = 3
        self.lay_objects = 4
        self.rebuild()

    def make_engagement(self):
        """Single construction point for the engagement.

        Both rebuild() and _escalate() go through here. When escalation built
        its own Engagement separately the two drifted -- escalation silently
        picked up different settings than the run it was supposed to be
        continuing, and reported a 'hit' from what was actually a different
        engagement. One constructor, one set of arguments.
        """
        return Engagement(range_km=self.args.threat_range,
                          loft=self.args.loft, objects=self.args.objects)

    def rebuild(self, keep_view=False):
        self.eng = self.make_engagement()
        self.sim_t = 0.0
        if not keep_view:
            self.zoom_intercept = False
        self.interceptor_path = self._interceptor_path()
        self.probe_paths = self._probe_paths()
        self.escalate_log = []
        if self.auto_escalate:
            self._escalate()

    def _escalate(self, max_k=8):
        """'On a miss, add one more vehicle' -- run live, with a stall guard.

        Escalation is allowed to continue only while it is actually buying
        something. When the added vehicle fails to improve the miss distance
        the loop stops and says so, because past that point every vehicle is
        flying out the same shared track error and landing in the same place
        (see --escalate). Without the guard this rule escalates forever at
        degraded track quality while changing nothing.
        """
        log = []
        while self.term_hit is False and INTERCEPTOR["kv_count"] < max_k:
            prev_miss = self.term_miss_m
            INTERCEPTOR["kv_count"] += 1
            self.eng = self.make_engagement()
            self.interceptor_path = self._interceptor_path()
            self.probe_paths = self._probe_paths()
            improved = (prev_miss is not None and self.term_miss_m is not None
                        and self.term_miss_m < prev_miss * 0.999)
            log.append((INTERCEPTOR["kv_count"], self.term_miss_m,
                        bool(self.term_hit), improved))
            if not self.term_hit and not improved:
                log.append(("stall", self.term_miss_m, False, False))
                break
        self.escalate_log = log

    def _interceptor_path(self):
        """Propagate the interceptor's own arc from burnout to seeker
        acquisition, two-body, matching the Lambert solution that produced it.
        Then SPLICE ON the actual terminal-homing trajectory from a real
        homing_run(), so the path this drives on screen is the same physics
        the text reports quote -- not an idealised arc that always rendezvous
        exactly, with hit or miss decided afterward by a coin flip.

        The seed is fixed per engagement (not re-drawn per frame), so the
        picture is reproducible: rebuild without changing parameters and you
        get the same outcome, same as pinning `patient_seed` elsewhere in this
        project's sibling models.

        Sets self.term_hit, self.term_miss_m, self.term_dv_used and
        self.term_cpa_t (the absolute sim time of closest approach -- where
        the drawn path actually ends, which is close to but not exactly
        sol['t_intercept'], because the guided path is not the idealised one).
        """
        s = self.eng.solution
        self.term_hit = None
        self.term_miss_m = None
        self.term_dv_used = None
        self.term_cpa_t = None
        self.term_kv_count = 1
        self.term_kv_paths = []
        self.term_all_miss_m = []
        if not s:
            return None

        vc = max(s["closing_speed"], 1.0)
        acq_s = self.eng.spec["kv_seeker_acq_km"] * 1000.0 / vc
        t_acq = s["t_intercept"] - acq_s

        r = self.eng.site_r.copy()
        v = s["v_required"].copy()
        pts, ts = [r.copy()], [s["t_burnout"]]
        t = s["t_burnout"]
        while t < t_acq:
            step = min(2.0, t_acq - t)
            r, v = rk4_step(r, v, step, use_j2=False)
            t += step
            pts.append(r.copy())
            ts.append(t)

        # Fly every kill vehicle in the salvo, sharing ONE track draw between
        # them (they come off one bus with one fire-control solution) while
        # each gets its own aimpoint bias and seeker noise. With kv_count=1
        # this is exactly the previous single-vehicle behaviour.
        rng = np.random.default_rng(self.seed_terminal())
        kvc = max(1, int(self.eng.spec.get("kv_count", 1)))
        sigma = self.eng.track_sigma
        off = rng.normal(0.0, sigma, 2)
        voff = rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)

        lethal = self.eng.spec["kv_lethal_radius_m"]
        runs = []
        for _i in range(kvc):
            miss, dv, trace = homing_run(s, self.eng.spec, sigma, rng,
                                         noise=True, record=True,
                                         shared_track=(off, voff))
            runs.append((miss, dv, trace))

        # The vehicle that actually gets closest is the one that decides the
        # engagement, and is the one the primary path follows.
        best = int(np.argmin([r[0] for r in runs]))
        self.term_miss_m = runs[best][0]
        self.term_dv_used = runs[best][1]
        self.term_hit = self.term_miss_m <= lethal
        self.term_kv_count = kvc
        self.term_all_miss_m = [r[0] for r in runs]

        def resample(trace):
            """Adaptive-dt trace onto a uniform grid, so downstream
            searchsorted lookups behave like the rest of interceptor_path."""
            tau = np.array([p[0] for p in trace])
            rel = np.array([p[1] for p in trace])
            n = max(2, int(tau[-1] / 0.25) + 1)
            tau_u = np.linspace(0.0, tau[-1], n)
            rel_u = np.column_stack([np.interp(tau_u, tau, rel[:, i])
                                     for i in range(3)])
            return tau_u, rel_u

        tau_u, rel_u = resample(runs[best][2])
        for tu, ru in zip(tau_u, rel_u):
            t_abs = t_acq + tu
            target_r, _ = self.eng.traj.at(t_abs)
            pts.append(target_r + ru)
            ts.append(t_abs)
        self.term_cpa_t = t_acq + tau_u[-1]

        # Secondary vehicles, stored as absolute paths for the close-up view.
        # Each carries its OWN hit flag: every vehicle in the salvo targets
        # independently, so the view must be able to show one striking while
        # the others fly past, rather than treating the salvo as a single
        # pass/fail event.
        self.term_kv_paths = []
        for i, (miss, dv, trace) in enumerate(runs):
            if i == best:
                continue
            tu_i, ru_i = resample(trace)
            p_i = np.array([self.eng.traj.at(t_acq + tu)[0] + ru
                            for tu, ru in zip(tu_i, ru_i)])
            self.term_kv_paths.append((p_i, t_acq + tu_i, miss, miss <= lethal))

        return np.array(pts), np.array(ts)

    def _probe_paths(self):
        """Propagate full ballistic arcs for every probe shot in the shot
        ladder. Each probe fires at max capability toward its PIP and
        coasts on a Keplerian arc. Returns a list of (positions, times,
        shot_dict) tuples for animation in the globe view."""
        e = self.eng
        if not e.shot_ladder:
            return []
        paths = []
        for shot in e.shot_ladder:
            if shot["shot_type"] != "probe":
                continue
            # Build the trajectory: fire from the site at max dv toward
            # the Lambert direction (or directly at the PIP if no Lambert
            # solution existed).
            r = np.array(e.site_r, dtype=float)
            if math.isinf(shot["dv_required"]):
                # No Lambert solution: fire straight at the PIP
                v_dir = shot["r_pip"] - e.site_r
                v_dir = v_dir / np.linalg.norm(v_dir)
            else:
                # Lambert direction from the stored solution
                # Re-solve to get v_req (not stored in the ladder dict)
                tof = shot["flight_time_s"]
                sol = lambert_best(e.site_r, shot["r_pip"], tof)
                if sol is None:
                    continue
                v_req = sol[0]
                v_dir = v_req - e.v_site
                v_dir = v_dir / np.linalg.norm(v_dir)
            v = e.v_site + e.dv_available * v_dir
            t_launch = shot.get("t_launch", e.t_detect + e.spec["launch_delay_s"] + e.spec["burn_time_s"])
            pts = [r.copy()]
            ts = [t_launch]
            t = t_launch
            t_end = shot["t_intercept"]
            dt = max(1.0, (t_end - t_launch) / 200.0)
            while t < t_end:
                step = min(dt, t_end - t)
                r, v = rk4_step(r, v, step, use_j2=False)
                t += step
                pts.append(r.copy())
                ts.append(t)
            paths.append((np.array(pts), np.array(ts), shot))
        return paths

    def seed_terminal(self):
        """Fixed per-engagement seed for the illustrative terminal run, so the
        collision view is reproducible rather than re-rolled every rebuild."""
        return (hash((round(self.args.threat_range, 3), round(self.args.loft, 3),
                     self.args.objects)) & 0xFFFFFFFF) or 1

    # --- main loop ---

    def _draw_demo_caption(self, s, rect):
        """Caption strip for the self-running tour."""
        pg = self.pg
        name = MODES[self.mode]
        cap = MODE_CAPTIONS.get(name, "")
        h = 62
        bar = pg.Rect(rect.x + 40, rect.bottom - h - 14, rect.w - 80, h)
        pg.draw.rect(s, (12, 18, 30), bar, border_radius=6)
        pg.draw.rect(s, COL_PIP, bar, 1, border_radius=6)
        left = self.demo_hold - self.demo_t
        self._text(s, f"DEMO  {self.mode+1}/{len(MODES)}  {name}"
                      f"   (next in {left:.0f}s -- D to stop, TAB to take over)",
                   (bar.x + 14, bar.y + 8), self.font_s, COL_PIP)
        self._text(s, cap, (bar.x + 14, bar.y + 28), self.font, COL_TEXT)
        # progress bar for the current slide
        pw = int((bar.w - 28) * min(1.0, self.demo_t / self.demo_hold))
        pg.draw.rect(s, (30, 42, 62), pg.Rect(bar.x + 14, bar.bottom - 9,
                                              bar.w - 28, 3))
        pg.draw.rect(s, COL_PIP, pg.Rect(bar.x + 14, bar.bottom - 9, pw, 3))

    def _demo_enter_mode(self):
        """Put each mode into the state that shows it off, when the tour
        arrives. Static panels need nothing; the 3D and terminal views only
        read well from a particular vantage or a particular instant."""
        name = MODES[self.mode]
        if name == "ENGAGEMENT":
            # Second half of the tour slide sits in the terminal close-up, so
            # the demo shows the collision rather than only the globe.
            self.zoom_intercept = False
        elif name == "INTERCEPTOR":
            self.explode = False
            self.section = False
            self.cam_v.az = math.radians(-32.0)
            self.cam_v.el = math.radians(14.0)
        self.scroll = 0

    def tick(self, dt):
        """One frame of simulation state advance, then draw.

        Factored out of run() so the selftest can exercise it. The launch
        crash this replaces -- a reference to a counter deleted with the code
        that used it -- lived here and survived a suite that rendered every
        view mode individually but never ran the loop that drives them. Any
        code reachable only from run() is code nothing tests.
        """
        if self.playing:
            self.sim_t += dt * self.sim_speed
            if self.sim_t > self.eng.traj.tof:
                self.sim_t = 0.0

        if self.demo:
            self.demo_t += dt
            name = MODES[self.mode]
            # In ENGAGEMENT the tour spends its second half in the terminal
            # close-up, and parks the clock just before closest approach so
            # the intercept actually plays rather than being scrubbed past.
            if name == "ENGAGEMENT" and self.term_cpa_t is not None:
                half = self.demo_hold * 0.5
                if self.demo_t >= half and not self.zoom_intercept:
                    self.zoom_intercept = True
                    self.sim_t = max(0.0, self.term_cpa_t - 6.0)
            if self.demo_t >= self.demo_hold:
                self.demo_t = 0.0
                self.mode = (self.mode + 1) % len(MODES)
                self._demo_enter_mode()
        self._draw()

    def run(self):
        pg = self.pg
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pg.event.get():
                if not self._event(ev):
                    running = False
            self.tick(dt)
            pg.display.flip()
        pg.quit()

    def _event(self, ev):
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.VIDEORESIZE:
            self.screen = pg.display.set_mode((ev.w, ev.h), pg.RESIZABLE)
        if ev.type == pg.KEYDOWN:
            k = ev.key
            if k in (pg.K_ESCAPE, pg.K_q):
                return False
            if k == pg.K_TAB:
                self.mode = (self.mode + 1) % len(MODES)
                self.scroll = 0
                self.demo = False
            elif pg.K_1 <= k <= pg.K_9 and (k - pg.K_1) < len(MODES):
                self.mode = k - pg.K_1
                self.scroll = 0
                self.demo = False
            elif k == pg.K_SPACE:
                self.playing = not self.playing
            elif k == pg.K_r:
                self.rebuild()
            elif k == pg.K_l:
                self.labels = not self.labels
            elif k == pg.K_g:
                self.grid = not self.grid
            elif k == pg.K_h:
                self.show_help = not self.show_help
            elif k == pg.K_d:
                # Start/stop the tour. Taking any other action (TAB, a mode
                # key) also stops it, so the demo never fights the user.
                self.demo = not self.demo
                self.demo_t = 0.0
                if self.demo:
                    self.playing = True
                    self._demo_enter_mode()
            elif k == pg.K_COMMA:
                self.sim_speed = max(1.0, self.sim_speed / 1.5)
            elif k == pg.K_PERIOD:
                self.sim_speed = min(400.0, self.sim_speed * 1.5)
            elif k == pg.K_UP:
                self.scroll = max(0, self.scroll - 2)
            elif k == pg.K_DOWN:
                self.scroll += 2
            elif k == pg.K_PAGEUP:
                self.scroll = max(0, self.scroll - 18)
            elif k == pg.K_PAGEDOWN:
                self.scroll += 18
            elif k == pg.K_HOME:
                self.scroll = 0
            elif k == pg.K_e and MODES[self.mode] == "INTERCEPTOR":
                self.explode = not self.explode
            elif k == pg.K_x and MODES[self.mode] == "INTERCEPTOR":
                self.section = not self.section
            elif k == pg.K_z and MODES[self.mode] == "ENGAGEMENT":
                self.zoom_intercept = not self.zoom_intercept
                if self.zoom_intercept and self.term_cpa_t is not None:
                    self.sim_t = max(0.0, self.term_cpa_t - 8.0)
            # --- context-sensitive parameter keys ---
            elif k == pg.K_m and MODES[self.mode] == "GUN":
                self.gun_mat = (self.gun_mat + 1) % len(MATERIAL_NAMES)
            elif k == pg.K_n and MODES[self.mode] == "GUN":
                self.gun_nose = 0.005 if self.gun_nose >= 0.06 else \
                    min(0.06, self.gun_nose * 1.6)
            elif k in (pg.K_MINUS, pg.K_EQUALS) and MODES[self.mode] == "GUN":
                step = 343.0 * (2.0 if k == pg.K_EQUALS else -2.0)
                self.gun_v = max(343.0, min(30000.0, self.gun_v + step))
            elif k in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                d = 1 if k == pg.K_RIGHTBRACKET else -1
                if MODES[self.mode] == "GUN":
                    self.gun_barrel = max(1.0, min(400.0,
                                                   self.gun_barrel + d * 2.0))
                elif MODES[self.mode] == "LAYERED":
                    self.lay_layers = max(1, min(8, self.lay_layers + d))
            elif k in (pg.K_k, pg.K_j) and MODES[self.mode] == "ENGAGEMENT":
                # Drive the salvo size live. Rebuilds the engagement because
                # kill-vehicle count changes the Pk computation, not just the
                # picture -- the number on screen must stay the number the
                # physics produced.
                d = 1 if k == pg.K_k else -1
                INTERCEPTOR["kv_count"] = max(1, min(8,
                                              INTERCEPTOR["kv_count"] + d))
                self.rebuild(keep_view=True)
            elif k == pg.K_m and MODES[self.mode] == "ENGAGEMENT":
                self.show_math = not self.show_math
            elif k == pg.K_a and MODES[self.mode] == "ENGAGEMENT":
                # Auto-escalation: on a missed terminal run, add a vehicle and
                # re-fly, until it hits or stops improving.
                self.auto_escalate = not self.auto_escalate
                if self.auto_escalate:
                    self.rebuild(keep_view=True)
            elif k in (pg.K_o, pg.K_p) and MODES[self.mode] == "LAYERED":
                d = 1 if k == pg.K_p else -1
                self.lay_objects = max(1, min(30, self.lay_objects + d))
        veh = MODES[self.mode] == "INTERCEPTOR"
        zoomed = MODES[self.mode] == "ENGAGEMENT" and self.zoom_intercept
        cam = self.cam_v if veh else (self.cam_z if zoomed else self.cam)
        lo, hi = (6000.0, 400000.0) if veh else (9e6, 140e6)
        if ev.type == pg.MOUSEBUTTONDOWN:
            if ev.button == 1:
                self.dragging = True
                self.last_mouse = ev.pos
            elif ev.button == 4:
                # The zoom view auto-frames cam_z.dist from true separation
                # every frame (see _engagement_zoom_view), so the wheel can't
                # drive dist directly -- it would be overwritten next frame,
                # exactly the bug this replaced. It drives a multiplier on
                # top of the auto-framing instead.
                if zoomed:
                    self.zoom_scale = max(0.03, self.zoom_scale / 1.15)
                else:
                    cam.dist = max(lo, cam.dist / 1.12)
                self.scroll = max(0, self.scroll - 3)
            elif ev.button == 5:
                if zoomed:
                    self.zoom_scale = min(25.0, self.zoom_scale * 1.15)
                else:
                    cam.dist = min(hi, cam.dist * 1.12)
                self.scroll += 3
        if ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        if ev.type == pg.MOUSEMOTION and self.dragging and self.last_mouse:
            dx = ev.pos[0] - self.last_mouse[0]
            dy = ev.pos[1] - self.last_mouse[1]
            cam.az += dx * 0.006
            cam.el = max(-1.4, min(1.4, cam.el + dy * 0.006))
            self.last_mouse = ev.pos
        return True

    # --- draw helpers ---

    def _text(self, surf, s, pos, font=None, col=COL_TEXT):
        surf.blit((font or self.font).render(s, True, col), pos)

    def _panel(self, surf, rect, title=None):
        pg = self.pg
        pg.draw.rect(surf, COL_PANEL, rect, border_radius=6)
        pg.draw.rect(surf, COL_GRID, rect, 1, border_radius=6)
        # Subtle top accent line for depth
        pg.draw.line(surf, (24, 34, 52), (rect.x + 2, rect.y + 1),
                     (rect.right - 2, rect.y + 1), 1)
        if title:
            self._text(surf, title, (rect.x + 12, rect.y + 8), self.font_b,
                       COL_ACCENT)

    def _aacircle(self, surf, col, cx, cy, r):
        """Anti-aliased filled circle with subtle dark rim."""
        pg = self.pg
        cx, cy, r = int(cx), int(cy), int(r)
        if r <= 0:
            return
        pg.draw.circle(surf, col, (cx, cy), r)
        if r > 1:
            pg.gfxdraw.aacircle(surf, cx, cy, r, col)
            rim = tuple(max(0, c - 40) for c in col)
            pg.gfxdraw.aacircle(surf, cx, cy, r, rim)

    def _glow_marker(self, surf, col, cx, cy, r):
        """Marker with a soft multi-ring glow halo."""
        pg = self.pg
        cx, cy, r = int(cx), int(cy), int(r)
        for gr, ga in ((r + 8, 28), (r + 5, 50), (r + 2, 80)):
            glow = tuple(min(255, int(c * ga / 255)) for c in col)
            pg.draw.circle(surf, glow, (cx, cy), gr, 1)
        self._aacircle(surf, col, cx, cy, r)

    def _aapolyline(self, surf, pts, col, width=2):
        """Anti-aliased polyline using gfxdraw aalines when width <= 2."""
        pg = self.pg
        if len(pts) < 2:
            return
        if width <= 2:
            pg.draw.aalines(surf, col, False, pts)
            if width == 2:
                pg.draw.aalines(surf, col, False,
                                [(x, y + 1) for x, y in pts])
        else:
            pg.draw.lines(surf, col, False, pts, width)
            pg.draw.aalines(surf, col, False, pts)

    def _glow_line(self, surf, pts, col, width=2):
        """Polyline with a dimmer, wider underlayer for a glow effect."""
        pg = self.pg
        if len(pts) < 2:
            return
        glow = tuple(max(0, int(c * 0.35)) for c in col)
        pg.draw.lines(surf, glow, False, pts, width + 2)
        pg.draw.lines(surf, col, False, pts, width)
        pg.draw.aalines(surf, col, False, pts)

    def _arrowhead(self, surf, col, p_from, p_to, size=10):
        """Draw an arrowhead at p_to pointing from p_from."""
        import math as _m
        dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
        ang = _m.atan2(dy, dx)
        a1 = ang + _m.radians(150)
        a2 = ang - _m.radians(150)
        pts = [p_to,
               (p_to[0] + size * _m.cos(a1), p_to[1] + size * _m.sin(a1)),
               (p_to[0] + size * _m.cos(a2), p_to[1] + size * _m.sin(a2))]
        self.pg.draw.polygon(surf, col, pts)
        self.pg.draw.aalines(surf, col, True, pts)

    def _draw(self):
        pg = self.pg
        s = self.screen
        s.fill(COL_BG)
        W, H = s.get_size()

        pg.draw.rect(s, COL_PANEL, pg.Rect(0, 0, W, 54))
        pg.draw.line(s, COL_GRID, (0, 54), (W, 54))
        pg.draw.line(s, (24, 34, 52), (0, 53), (W, 53), 1)
        self._text(s, "ICBMI", (16, 12), self.font_t, COL_ACCENT)

        # Right-hand HUD is laid out FIRST so the tab strip knows how much room
        # it actually has. Previously the tabs ran at a fixed pitch and simply
        # overran the HUD on a narrow window.
        m = self.eng.meta
        hud = (f"threat {m['range_km']:,.0f} km   apogee {m['apogee_km']:,.0f} km"
               f"   T+{self.sim_t/60:5.1f} min   x{self.sim_speed:.0f}   "
               f"{'PLAY' if self.playing else 'PAUSE'}")
        hud_x = W - self.font.size(hud)[0] - 16
        self._text(s, hud, (hud_x, 20), self.font, COL_DIM)
        icon_x, icon_y = hud_x - 22, 22
        if self.playing:
            pg.draw.polygon(s, COL_INTER,
                            [(icon_x, icon_y), (icon_x + 8, icon_y + 5),
                             (icon_x, icon_y + 10)])
        else:
            pg.draw.rect(s, COL_WARN, pg.Rect(icon_x, icon_y, 3, 10))
            pg.draw.rect(s, COL_WARN, pg.Rect(icon_x + 5, icon_y, 3, 10))

        tab_x, tab_y = 90, 10
        tab_limit = icon_x - 30
        for i, mode_name in enumerate(MODES):
            label = mode_name.title()
            tw = self.font_s.size(label)[0] + 16
            if tab_x + tw > tab_limit:               # out of room: stop
                break
            if i == self.mode:
                r = pg.Rect(tab_x, tab_y, tw, 20)
                pg.draw.rect(s, (28, 40, 60), r, border_radius=4)
                pg.draw.rect(s, COL_ACCENT, r, 1, border_radius=4)
                self._text(s, label, (tab_x + 8, tab_y + 3), self.font_s,
                           COL_ACCENT)
            else:
                self._text(s, label, (tab_x + 8, tab_y + 3), self.font_s,
                           COL_DIM)
            tab_x += tw + 4

        # Body ends where the status bar begins -- these disagreed by 6 px, so
        # panels sized to the body could tuck under the status bar.
        body = pg.Rect(0, 54, W, H - 54 - 26)
        try:
            getattr(self, "_mode_" + MODES[self.mode].lower())(s, body)
        except Exception as ex:                      # never kill the viewer
            self._text(s, f"render error in {MODES[self.mode]}: "
                          f"{type(ex).__name__}: {ex}",
                       (24, body.y + 24), self.font, COL_WARN)
            self._text(s, "the rest of the program is unaffected; press TAB",
                       (24, body.y + 46), self.font_s, COL_DIM)

        if self.demo:
            self._draw_demo_caption(s, body)

        # Status bar: controls for THIS mode, then the global ones.
        pg.draw.rect(s, COL_PANEL, pg.Rect(0, H - 26, W, 26))
        pg.draw.line(s, COL_GRID, (0, H - 26), (W, H - 26))
        pg.draw.line(s, (24, 34, 52), (0, H - 25), (W, H - 25), 1)
        local = MODE_CONTROLS.get(MODES[self.mode], "")
        self._text(s, local, (14, H - 22), self.font_s, COL_PIP)
        gl = "TAB mode   SPACE play   ,/. speed   R rebuild   D demo   H help   ESC quit"
        gx = W - self.font_s.size(gl)[0] - 14
        if gx > 14 + self.font_s.size(local)[0] + 24:
            self._text(s, gl, (gx, H - 22), self.font_s, COL_DIM)

        if self.show_help:
            self._draw_help(s, body)

    def _draw_help(self, s, rect):
        pg = self.pg
        r = pg.Rect(rect.centerx - 370, rect.centery - 210, 740, 420)
        self._panel(s, r, "HELP")
        lines = [
            "",
            "  PRESS  D  FOR A SELF-RUNNING TOUR OF EVERY MODE",
            "  (captions explain each view; TAB or a number key takes over)",
            "",
            "  1 ENGAGEMENT      globe view of both arcs and the intercept",
            "  2 INTERCEPTOR     booster and kill vehicle, drawn to scale",
            "  3 GEOMETRY        closing velocity and miss distribution",
            "  4 SENSORS         radar range equation and horizon limits",
            "  5 TIMELINE        every delay in the engagement, to scale",
            "  6 BATTLESPACE     window and shot count vs threat range",
            "  7 DISCRIMINATION  object count vs interceptor inventory",
            "  8 GUN             numeric assessment only -- see note below",
            "  9 LAYERED         independence vs volume of fire",
            "    PHYSICS         the equations, with live values substituted",
            "    ABOUT           scope, provenance, and omissions",
            "    (TAB cycles through all modes; 1-9 jump directly)",
            "",
            "  The status bar always shows the keys that do something in the",
            "  mode you are actually looking at.",
            "",
            "  ENGAGEMENT:    Z terminal close-up -- the real simulated",
            "                 collision, not a globe-scale dot.",
            "                 J / K salvo size    A auto-escalate on a miss",
            "                 M live intercept-math panel",
            "  INTERCEPTOR:   E exploded   X section cut   L labels",
            "  GUN mode:      M material   - / = velocity   [ / ] barrel   N nose",
            "  LAYERED mode:  [ / ] independent sites       O / P objects",
            "",
            "  GUN mode is numbers and pass/fail margins only, deliberately --",
            "  no rendered gun or projectile geometry. A dimensioned drawing of",
            "  a weapon is a different thing from a critique of one, and this",
            "  project only does the second.",
            "",
            "  The threat enters this model as a ballistic arc derived from",
            "  ground range alone. No threat-vehicle design is modelled and",
            "  none is needed -- gravity sets the trajectory, which is the",
            "  whole reason midcourse intercept is a tractable kinematics",
            "  problem rather than a guessing game.",
            "",
            "  Press H to close.",
        ]
        for i, ln in enumerate(lines):
            self._text(s, ln, (r.x + 16, r.y + 42 + i * 19), self.font,
                       COL_TEXT if ln.strip()[:1].isdigit() else COL_DIM)

    # --- globe ---

    def _sphere(self, s, rect, nlat=12, nlon=24):
        pg = self.pg
        cam = self.cam
        c = cam.project(np.zeros(3), rect)
        if c:
            focal = min(rect.w, rect.h) * 1.05
            rad = focal * R_E / c[2]
            if 2 < rad < 20000:
                cx, cy = int(c[0]), int(c[1])
                irad = int(rad)
                # Atmospheric glow: concentric rings fading outward
                for gr, ga in ((irad + 14, 18), (irad + 10, 30),
                               (irad + 7, 50), (irad + 4, 80)):
                    glow = tuple(min(255, int(col * ga / 255))
                                 for col in COL_ATMOS)
                    pg.draw.circle(s, glow, (cx, cy), gr, 1)
                # Earth disc with subtle limb darkening: draw concentric
                # rings from outer edge inward, brightening toward centre
                for i in range(irad, 0, -2):
                    t = 1.0 - i / irad
                    shade = int(40 * t)
                    col = (min(255, COL_EARTH[0] + shade),
                           min(255, COL_EARTH[1] + shade),
                           min(255, COL_EARTH[2] + shade))
                    pg.draw.circle(s, col, (cx, cy), i)
                # Karman line
                karman_r = int(rad * (1.0 + KARMAN_M / R_E))
                pg.draw.circle(s, COL_LIMB, (cx, cy), karman_r, 1)
                # Anti-aliased rim
                pg.gfxdraw.aacircle(s, cx, cy, irad, COL_LIMB)
        if not self.grid:
            return
        cd = cam.cam_dir()

        def strand(pts):
            run = []
            for p in pts:
                q = cam.project(p, rect)
                if q and float(np.dot(p, cd)) > 0:
                    run.append((q[0], q[1], q[2]))
                else:
                    if len(run) > 1:
                        for i in range(len(run) - 1):
                            d = run[i][2]
                            fade = max(0.25, min(1.0, 1.0 - d / 6e7))
                            gcol = tuple(min(255, int(c * fade))
                                         for c in COL_GRID)
                            pg.draw.line(s, gcol,
                                         (run[i][0], run[i][1]),
                                         (run[i + 1][0], run[i + 1][1]), 1)
                    run = []
            if len(run) > 1:
                for i in range(len(run) - 1):
                    d = run[i][2]
                    fade = max(0.25, min(1.0, 1.0 - d / 6e7))
                    gcol = tuple(min(255, int(c * fade)) for c in COL_GRID)
                    pg.draw.line(s, gcol,
                                 (run[i][0], run[i][1]),
                                 (run[i + 1][0], run[i + 1][1]), 1)

        for i in range(1, nlat):
            lat = -math.pi / 2 + math.pi * i / nlat
            strand([np.array([R_E*math.cos(lat)*math.cos(2*math.pi*j/nlon),
                              R_E*math.cos(lat)*math.sin(2*math.pi*j/nlon),
                              R_E*math.sin(lat)]) for j in range(nlon + 1)])
        for j in range(nlon):
            lon = 2 * math.pi * j / nlon
            strand([np.array([R_E*math.cos(-math.pi/2 + math.pi*i/(nlat*2))*math.cos(lon),
                              R_E*math.cos(-math.pi/2 + math.pi*i/(nlat*2))*math.sin(lon),
                              R_E*math.sin(-math.pi/2 + math.pi*i/(nlat*2))])
                    for i in range(nlat * 2 + 1)])

    def _polyline(self, s, pts3, rect, col, width=2):
        run = []
        for p in pts3:
            q = self.cam.project(p, rect)
            if q:
                run.append((q[0], q[1]))
        if len(run) > 1:
            self._aapolyline(s, run, col, width)

    def _marker(self, s, p3, rect, col, r=5, label=None):
        q = self.cam.project(p3, rect)
        if not q:
            return
        self._glow_marker(s, col, q[0], q[1], r)
        if label and self.labels:
            self._text(s, label, (q[0] + r + 6, q[1] - 8), self.font_s, col)

    def _glow_polyline(self, s, pts3, rect, col, width=2):
        """3D polyline with a dimmer, wider underlayer for a glow effect."""
        run = []
        for p in pts3:
            q = self.cam.project(p, rect)
            if q:
                run.append((q[0], q[1]))
        if len(run) > 1:
            self._glow_line(s, run, col, width)

    # --- MODE 1 ---

    def _mode_engagement(self, s, rect):
        pg = self.pg
        e = self.eng
        view = pg.Rect(rect.x, rect.y, rect.w - 400, rect.h)
        side = pg.Rect(rect.right - 392, rect.y + 8, 384, rect.h - 16)

        if self.zoom_intercept and e.solution:
            self._engagement_zoom_view(s, view, e)
        else:
            self._engagement_globe_view(s, view, e)

        self._panel(s, side, "ENGAGEMENT STATE")
        y = [side.y + 40]

        def row(k, v, col=COL_TEXT):
            self._text(s, k, (side.x + 14, y[0]), self.font_s, COL_DIM)
            self._text(s, v, (side.x + 212, y[0]), self.font_s, col)
            y[0] += 19

        def head(t):
            y[0] += 10
            self._text(s, t, (side.x + 14, y[0]), self.font, COL_ACCENT)
            y[0] += 22

        m = e.meta
        row("threat range", f"{m['range_km']:,.0f} km")
        row("apogee", f"{m['apogee_km']:,.0f} km")
        row("burnout speed", f"{m['v_burnout_ms']/1000:.2f} km/s")
        row("total flight", f"{m['total_time_s']/60:.1f} min")
        row("reentry speed", f"{m['entry_speed_ms']/1000:.2f} km/s")
        row("impact speed", f"{m['impact_speed_ms']/1000:.2f} km/s")

        head("DETECTION")
        row("EW radar limit", f"{e.rmax_ew/1000:,.0f} km")
        row("first detect", "never" if e.t_detect is None
            else f"T+{e.t_detect/60:.1f} min")

        if e.solution:
            sol = e.solution
            head("FIRE CONTROL")
            row("launch", f"T+{sol['t_launch']/60:.1f} min")
            row("intercept", f"T+{sol['t_intercept']/60:.1f} min")
            row("intercept alt", f"{sol['alt_km']:,.0f} km")
            row("dv required", f"{sol['dv_required']/1000:.2f} km/s")
            row("energy margin", f"{sol['margin_ms']/1000:+.2f} km/s",
                COL_INTER if sol["margin_ms"] > 0 else COL_WARN)
            # Energy margin bar: green if positive, red if negative
            em_x, em_y = side.x + 14, y[0]
            em_w, em_h = side.w - 28, 8
            pg.draw.rect(s, COL_DARK, pg.Rect(em_x, em_y, em_w, em_h))
            pg.draw.rect(s, COL_GRID, pg.Rect(em_x, em_y, em_w, em_h), 1)
            # Zero line at centre
            zero_x = em_x + em_w // 2
            pg.draw.line(s, COL_DIM, (zero_x, em_y - 1), (zero_x, em_y + em_h + 1), 1)
            margin_frac = max(-1.0, min(1.0, sol["margin_ms"] / INTERCEPTOR["burnout_v_ms"]))
            if margin_frac >= 0:
                fw = int(margin_frac * em_w / 2)
                pg.draw.rect(s, COL_INTER, pg.Rect(zero_x, em_y, fw, em_h))
                hl = tuple(min(255, c + 30) for c in COL_INTER)
                pg.draw.line(s, hl, (zero_x, em_y), (zero_x + fw, em_y), 1)
            else:
                fw = int(-margin_frac * em_w / 2)
                pg.draw.rect(s, COL_WARN, pg.Rect(zero_x - fw, em_y, fw, em_h))
                hl = tuple(min(255, c + 30) for c in COL_WARN)
                pg.draw.line(s, hl, (zero_x - fw, em_y), (zero_x, em_y), 1)
            y[0] += 14
            row("closing speed", f"{sol['closing_speed']/1000:.2f} km/s")

            head("TERMINAL")
            row("track error", f"{e.track_sigma:,.0f} m")
            row("median miss", f"{e.pk['miss_median']:.2f} m")
            row("divert p95", f"{e.pk['dv_p95']:.0f} m/s",
                COL_WARN if e.pk["dv_p95"] > INTERCEPTOR["kv_divert_dv_ms"] * 0.9
                else COL_TEXT)
            pk_val = e.pk["pk"]
            kvc = int(e.spec.get("kv_count", 1))
            if kvc > 1:
                row("kill vehicles", f"{kvc}  (salvo)")
            # pct_mc rather than a raw percentage: an all-hit Monte Carlo is a
            # bound set by the sample size, not a demonstration of certainty.
            row("SALVO Pk" if kvc > 1 else "SINGLE-SHOT Pk", pct_mc(e.pk),
                COL_INTER if pk_val > 0.7 else COL_WARN)
            # Visual Pk bar in the side panel
            pk_col = COL_INTER if pk_val > 0.7 else COL_WARN if pk_val > 0.1 else COL_THREAT
            bk_x, bk_y, bk_w, bk_h = side.x + 14, y[0], side.w - 28, 10
            pg.draw.rect(s, COL_DARK, pg.Rect(bk_x, bk_y, bk_w, bk_h))
            pk_fill = int(pk_val * bk_w)
            if pk_fill > 0:
                pg.draw.rect(s, pk_col, pg.Rect(bk_x, bk_y, pk_fill, bk_h))
                hl = tuple(min(255, c + 30) for c in pk_col)
                pg.draw.line(s, hl, (bk_x, bk_y), (bk_x + pk_fill, bk_y), 1)
            pg.draw.rect(s, COL_GRID, pg.Rect(bk_x, bk_y, bk_w, bk_h), 1)
            y[0] += 16

            head("INVENTORY")
            row("objects on track", f"{e.objects}")
            row("shots available", f"{e.shot_opportunities}")
            need = e.interceptors_for_leakage(0.01)
            row("for 99% intercept", str(need) if need else "unreachable",
                COL_WARN if (need or 99) > 4 else COL_INTER)

            if e.shot_ladder and len(e.shot_ladder) > 1:
                head("SHOT LADDER")
                n_early = sum(1 for sh in e.shot_ladder
                              if sh["shot_type"] == "early")
                n_probe = sum(1 for sh in e.shot_ladder
                              if sh["shot_type"] == "probe")
                n_salvo = sum(1 for sh in e.shot_ladder
                              if sh["shot_type"] == "salvo")
                n_trail = sum(1 for sh in e.shot_ladder
                              if sh["shot_type"] == "trailing")
                total_kvs = sum(sh["kv_count"] for sh in e.shot_ladder)
                row("early shots", f"{n_early}  (500 km spacing)")
                row("probe shots", f"{n_probe}  (fire & hope)")
                row("main salvo", f"{n_salvo}")
                row("trailing shots", f"{n_trail}  (single only)")
                row("total KVs", f"{total_kvs}")
                leak = layered_leakage(e.shot_ladder, e.objects)
                row("layered leakage", f"{leak*100:.3f}%",
                    COL_INTER if leak < 0.05 else COL_WARN)
        else:
            head("FIRE CONTROL")
            self._text(s, "NO FEASIBLE INTERCEPT", (side.x + 14, y[0]),
                       self.font, COL_WARN)

    # --- engagement sub-views ---

    def _engagement_globe_view(self, s, view, e):
        """Whole-flight globe view. Reads the SAME terminal state the
        close-up and the text reports use -- self.term_hit, term_miss_m,
        term_cpa_t, term_kv_paths -- so a hit or a miss, and the full salvo,
        are visible here too rather than only after pressing Z. At this
        camera distance a metre-scale miss is sub-pixel, which is correct
        perspective rather than something to fake; the outcome is still
        shown as colour and text, not left silent."""
        pg = self.pg
        self._sphere(s, view)
        self._polyline(s, e.traj.r[::4], view, (118, 48, 44), 2)
        idx = int(np.searchsorted(e.traj.t, self.sim_t))
        threat_alive = not (self.term_cpa_t is not None
                            and self.sim_t >= self.term_cpa_t
                            and self.term_hit)
        if idx > 2 and threat_alive:
            self._glow_polyline(s, e.traj.r[:idx:3], view, COL_THREAT, 3)

        cpa_reached = (self.term_cpa_t is not None
                       and self.sim_t >= self.term_cpa_t)
        out_col = COL_PIP if self.term_hit else COL_WARN

        if self.interceptor_path is not None:
            ip, it = self.interceptor_path
            self._polyline(s, ip[::2], view, (36, 92, 66), 2)
            if self.sim_t > it[0]:
                j = min(int(np.searchsorted(it, self.sim_t)), len(ip) - 1)
                if j > 2:
                    trail_col = out_col if cpa_reached else COL_INTER
                    self._glow_polyline(s, ip[:j], view, trail_col, 3)
                    if not (cpa_reached and self.term_hit):
                        self._marker(s, ip[j], view, trail_col, 5,
                                     "interceptor")

        # Probe shots: animated ballistic arcs firing at each 500 km
        # increment before the main salvo. These are "fire and hope"
        # attempts -- the interceptor launches at max capability toward
        # each point even though it can't reach the PIP.
        if self.probe_paths:
            for pp, pt, shot in self.probe_paths:
                if self.sim_t < pt[0]:
                    continue
                j = min(int(np.searchsorted(pt, self.sim_t)), len(pp) - 1)
                if j < 2:
                    continue
                col = (120, 60, 40)  # dark red-brown for probes
                self._polyline(s, pp[:j:2], view, col, 1)
                self._marker(s, pp[j], view, col, 3)

        # Rest of the salvo. Individual vehicles are metres apart, which is
        # invisible at this scale and honestly should be -- but the salvo
        # still needs to exist on screen, not just in the terminal close-up.
        for p_i, t_i, miss_i, hit_i in self.term_kv_paths:
            if self.sim_t < t_i[0]:
                continue
            j = min(int(np.searchsorted(t_i, self.sim_t)), len(p_i) - 1)
            if j < 2:
                continue
            if cpa_reached and hit_i:
                continue
            vcol = (out_col if cpa_reached else COL_INTER)
            self._polyline(s, p_i[:j:2], view, vcol, 1)

        if cpa_reached:
            n_hit = sum(1 for m in self.term_all_miss_m
                        if m <= e.spec["kv_lethal_radius_m"])
            outcome = (f"HIT -- {n_hit}/{self.term_kv_count} vehicle"
                      f"{'s' if self.term_kv_count > 1 else ''} within "
                      f"lethal radius" if self.term_hit else
                      f"MISS -- cleared by {self.term_miss_m:,.1f} m")
            self._text(s, f"{outcome}   (Z for the collision itself)",
                       (view.x + 16, view.y + 8), self.font_b, out_col)
        else:
            self._text(s, "press Z to zoom to intercept and see the collision",
                       (view.x + 16, view.y + 8), self.font_s, COL_DIM)

        # Probe shot counter: show how many fire-and-hope probes are
        # active or have been fired at this point in the timeline.
        if self.probe_paths and not cpa_reached:
            n_active = 0
            n_done = 0
            for pp, pt, shot in self.probe_paths:
                if self.sim_t < pt[0]:
                    continue
                if self.sim_t >= pt[-1]:
                    n_done += 1
                else:
                    n_active += 1
            if n_active > 0 or n_done > 0:
                probe_txt = f"PROBES: {n_active} in flight, {n_done} spent"
                self._text(s, probe_txt,
                           (view.x + 16, view.bottom - 24),
                           self.font_s, (120, 60, 40))

        self._marker(s, e.site_r, view, COL_SENSOR, 6, "interceptor site")
        self._marker(s, e.ew_site_r, view, (86, 120, 190), 5, "EW radar")
        if e.rmax_ew and e.rmax_ew > 0:
            ring_pts = []
            for a in range(0, 361, 6):
                ang = math.radians(a)
                offset = np.array([e.rmax_ew * math.cos(ang),
                                   e.rmax_ew * math.sin(ang), 0.0])
                rp = e.ew_site_r + offset
                r_norm = np.linalg.norm(rp)
                if r_norm > R_E * 0.3:
                    rp = rp * (R_E * 1.01 / r_norm)
                ring_pts.append(rp)
            self._polyline(s, ring_pts, view, (50, 70, 110), 1)
        if e.solution:
            self._marker(s, e.solution["r_pip"], view, COL_PIP, 7,
                         f"PIP  {e.solution['alt_km']:,.0f} km")
        # Shot ladder markers: early/trailing shots as small diamonds,
        # salvo as a larger highlighted marker.
        if e.shot_ladder:
            for shot in e.shot_ladder:
                if shot["shot_type"] == "salvo":
                    continue  # already drawn as PIP
                q = self.cam.project(shot["r_pip"], view)
                if not q:
                    continue
                sx, sy = q[0], q[1]
                if shot["shot_type"] == "probe":
                    col = (120, 60, 40)   # dark red-brown: fire & hope
                elif shot["shot_type"] == "early":
                    col = (60, 140, 80)   # green: feasible early
                else:
                    col = (140, 100, 50)  # brown: trailing
                sz = 4 if shot["kv_count"] > 1 else 3
                pg.draw.polygon(s, col, [(sx, sy - sz), (sx + sz, sy),
                                         (sx, sy + sz), (sx - sz, sy)])
        if threat_alive:
            self._marker(s, e.traj.r[min(idx, len(e.traj.r) - 1)], view,
                         COL_THREAT, 6, "threat")
        if idx < len(e.traj.r) and threat_alive:
            tpos = e.traj.r[min(idx, len(e.traj.r) - 1)]
            talt = (np.linalg.norm(tpos) - R_E) / 1000.0
            if idx < len(e.traj.v):
                tspd = np.linalg.norm(e.traj.v[min(idx, len(e.traj.v) - 1)]) / 1000.0
            else:
                tspd = 0.0
            overlay_x = view.x + 16
            overlay_y = view.bottom - 60
            self._text(s, f"threat alt {talt:,.0f} km   speed {tspd:.2f} km/s",
                       (overlay_x, overlay_y), self.font_s, COL_THREAT)
            apogee = e.meta["apogee_km"]
            ab_x = overlay_x
            ab_y = overlay_y + 18
            ab_w = 160
            ab_h = 6
            pg.draw.rect(s, COL_DARK, pg.Rect(ab_x, ab_y, ab_w, ab_h))
            pg.draw.rect(s, COL_GRID, pg.Rect(ab_x, ab_y, ab_w, ab_h), 1)
            ab_fill = int(min(1.0, max(0.0, talt / max(1.0, apogee))) * ab_w)
            if ab_fill > 0:
                pg.draw.rect(s, COL_THREAT, pg.Rect(ab_x, ab_y, ab_fill, ab_h))
            self._text(s, f"apogee {apogee:,.0f} km",
                       (ab_x + ab_w + 8, ab_y - 2), self.font_s, COL_DIM)

    def _path_at(self, ip, it, t):
        """Position and velocity on a (positions, times) path array at time t,
        linearly interpolated and clamped to the ends. Past the last sample
        this holds the final position and the closing segment's velocity --
        which is exactly the freeze-frame behaviour wanted at closest
        approach: the picture stops where the real simulated path stopped."""
        t = max(float(it[0]), min(float(t), float(it[-1])))
        j = int(np.searchsorted(it, t) - 1)
        j = max(0, min(j, len(it) - 2))
        span = it[j + 1] - it[j]
        f = 0.0 if span <= 0 else (t - it[j]) / span
        pos = ip[j] + f * (ip[j + 1] - ip[j])
        vel = (ip[j + 1] - ip[j]) / max(1e-6, span)
        return pos, vel

    def _engagement_zoom_view(self, s, view, e):
        """Close-up 3D view of the intercept, driven by the SAME simulated
        terminal trajectory _interceptor_path() already spliced in -- not a
        separate hit/miss roll. Whether the two objects visually meet or pass
        each other is read directly off that path, so the picture and the
        numeric reports can never disagree."""
        pg = self.pg
        sol = e.solution
        cam = self.cam_z

        threat_pos, threat_vel = e.traj.at(self.sim_t)
        int_pos, int_vel = (None, None)
        if self.interceptor_path is not None:
            ip, it = self.interceptor_path
            int_pos, int_vel = self._path_at(ip, it, self.sim_t)

        sep = (float(np.linalg.norm(int_pos - threat_pos))
               if int_pos is not None else 4e5)
        cam.center = threat_pos.copy()
        cam.dist = max(4.0, min(5e8, sep * 2.4 * self.zoom_scale))

        s.fill(COL_BG, view)
        earth_proj = cam.project(np.zeros(3), view)
        if earth_proj and earth_proj[2] > 0:
            focal = min(view.w, view.h) * 1.05
            rad = focal * R_E / earth_proj[2]
            if 2 < rad < 50000 and earth_proj[1] - rad < view.bottom:
                for gr, ga in ((int(rad) + 14, 18), (int(rad) + 10, 30),
                               (int(rad) + 7, 50), (int(rad) + 4, 80)):
                    glow = tuple(min(255, int(col * ga / 255))
                                 for col in COL_ATMOS)
                    pg.draw.circle(s, glow, (int(earth_proj[0]),
                                             int(earth_proj[1])), gr, 1)

        if self._kv_mesh is None:
            self._kv_mesh = build_kv_mesh(INTERCEPTOR)
        if self._rv_mesh is None:
            self._rv_mesh = build_threat_rv_mesh()

        light = np.array([0.42, 0.66, 0.62])
        light /= np.linalg.norm(light)
        Rc = cam.matrix()
        polys = []
        MM_TO_M = 0.001    # the only honest conversion -- no visibility inflation

        cpa_reached = (self.term_cpa_t is not None
                       and self.sim_t >= self.term_cpa_t)
        # The target is destroyed only if some vehicle actually reached inside
        # the lethal radius in the simulation. Vehicles that missed keep
        # flying -- each shot targets independently, so a salvo can show one
        # striking while the rest sail past, which is what actually happens.
        killed = cpa_reached and self.term_hit
        if not killed:
            self._collect_object_polys(self._rv_mesh, threat_pos, threat_vel,
                                       MM_TO_M, Rc, cam, view, light, polys)
        # Primary vehicle: hidden only once IT has struck.
        if int_pos is not None and not (cpa_reached and self.term_hit):
            self._collect_object_polys(self._kv_mesh, int_pos, int_vel,
                                       MM_TO_M, Rc, cam, view, light, polys)

        # Remaining vehicles of the salvo, each on its own real homing path
        # and its own outcome.
        for p_i, t_i, miss_i, hit_i in self.term_kv_paths:
            if self.sim_t < t_i[0]:
                continue
            if cpa_reached and hit_i:
                continue          # this one struck; it is gone too
            pos_i, vel_i = self._path_at(p_i, t_i, self.sim_t)
            self._collect_object_polys(self._kv_mesh, pos_i, vel_i,
                                       MM_TO_M, Rc, cam, view, light, polys)

        polys.sort(key=lambda t: -t[0])
        for _, scr, col, edge in polys:
            if len(scr) >= 3:
                pg.draw.polygon(s, col, scr)
                try:
                    pg.gfxdraw.aapolygon(s,
                        [(int(x), int(y)) for x, y in scr], edge)
                except Exception:
                    pass

        if not cpa_reached:
            self._draw_approach_trails(s, view, cam, e)

        if cpa_reached:
            if self.term_hit:
                self._draw_collision(s, view, cam, threat_pos)
            else:
                self._draw_flyby(s, view, cam, int_pos, threat_pos)

        if self.show_math:
            self._draw_intercept_math(s, view, e, sol, int_pos, int_vel,
                                      threat_pos, threat_vel)

        self._text(s, "TERMINAL INTERCEPT  (Z globe   scroll zoom   drag orbit"
                      "   J/K salvo   A auto-escalate   M math)",
                   (view.x + 16, view.y + 8), self.font_b, COL_ACCENT)
        salvo_note = ""
        if self.term_kv_count > 1:
            n_hit = sum(1 for m in self.term_all_miss_m
                        if m <= self.eng.spec["kv_lethal_radius_m"])
            salvo_note = (f"   salvo {self.term_kv_count} KV "
                          f"({n_hit} within lethal radius)")
        self._text(s, f"separation {sep:,.1f} m   view distance "
                      f"{cam.dist:,.0f} m   closing "
                      f"{sol['closing_speed']/1000:.2f} km/s{salvo_note}",
                   (view.x + 16, view.y + 30), self.font_s, COL_DIM)

        # Escalation ledger: what adding vehicles actually bought. Shown even
        # when it bought nothing, because that is the informative case.
        if self.auto_escalate:
            ey = view.y + 96
            self._text(s, "AUTO-ESCALATE ON  (A to disable)",
                       (view.x + 16, ey), self.font_s, COL_PIP)
            ey += 17
            if not self.escalate_log:
                self._text(s, "   first salvo hit -- no escalation needed",
                           (view.x + 16, ey), self.font_s, COL_INTER)
            for entry in self.escalate_log:
                if entry[0] == "stall":
                    self._text(s, f"   STALLED at {entry[1]:,.0f} m -- extra "
                                  f"vehicles land in the same place",
                               (view.x + 16, ey), self.font_s, COL_WARN)
                else:
                    kk, miss, hit, improved = entry
                    col = COL_INTER if hit else (COL_TEXT if improved
                                                 else COL_DIM)
                    self._text(s, f"   +1 -> {kk} KV: miss {miss:,.2f} m"
                                  f"{'  HIT' if hit else ''}",
                               (view.x + 16, ey), self.font_s, col)
                ey += 17

        if cpa_reached:
            lr = self.eng.spec["kv_lethal_radius_m"]
            if self.term_hit:
                ke = 0.5 * self.eng.spec["kv_mass_kg"] * sol["closing_speed"] ** 2
                self._text(s, f"HIT -- contact within lethal radius "
                              f"({self.term_miss_m:.2f} m <= {lr:.2f} m)  "
                              f"kinetic impact {ke/4.184e9:.2f} t TNT equiv.",
                           (view.x + 16, view.y + 50), self.font, COL_PIP)
            else:
                self._text(s, f"MISS -- cleared by {self.term_miss_m:.1f} m "
                              f"(lethal radius {lr:.2f} m)  divert used "
                              f"{self.term_dv_used:.0f} of "
                              f"{self.eng.spec['kv_divert_dv_ms']:.0f} m/s",
                           (view.x + 16, view.y + 50), self.font, COL_WARN)
        else:
            dt_to_cpa = ((self.term_cpa_t - self.sim_t)
                        if self.term_cpa_t is not None else float("nan"))
            self._text(s, f"T-{dt_to_cpa:.1f} s to closest approach",
                       (view.x + 16, view.y + 50), self.font, COL_TEXT)

    def _collect_object_polys(self, mesh, pos, vel, scale, Rc, cam, view,
                              light, polys):
        """Collect rendered polygons from a mesh placed at pos, oriented along
        vel, converted from its native millimetres to scene metres by `scale`.
        `scale` is always the true mm->m factor (0.001) -- there is no
        visibility multiplier here. What makes small objects visible at
        distance is camera framing, not resizing them."""
        v_norm = np.linalg.norm(vel)
        z_axis = vel / v_norm if v_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.99 \
            else np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])

        for p in mesh:
            v = p["verts"].copy()
            v_world = (v @ R.T) * scale + pos
            q = (v_world - cam.center) @ Rc.T
            q[:, 2] += cam.dist
            for f in p["faces"]:
                pts = q[list(f)]
                if np.any(pts[:, 2] <= 1e-2):
                    continue
                wpts = v_world[list(f)]
                n = np.cross(wpts[1] - wpts[0], wpts[2] - wpts[0])
                ln = np.linalg.norm(n)
                if ln < 1e-12:
                    continue
                n = n / ln
                diff = max(0.0, abs(float(np.dot(n, light))))
                sh = 0.22 + 0.58 * diff + 0.20 * (1.0 - diff) ** 2
                col = tuple(min(255, int(c * sh)) for c in p["color"])
                edge = tuple(max(0, int(c * 0.55)) for c in col)
                focal = min(view.w, view.h) * 1.05
                scr = [(view.centerx + focal * a[0] / a[2],
                        view.centery - focal * a[1] / a[2]) for a in pts]
                polys.append((float(pts[:, 2].mean()), scr, col, edge))

    def _draw_approach_trails(self, s, view, cam, e):
        """Short fading trails behind both objects on the way in."""
        pg = self.pg

        def trail(pts, col):
            run = []
            for p in pts:
                q = cam.project(p, view)
                if q:
                    run.append((q[0], q[1]))
            for i in range(len(run) - 1):
                fade = max(0.2, 1.0 - i / max(1, len(run)))
                tc = tuple(min(255, int(c * fade)) for c in col)
                pg.draw.aaline(s, tc, run[i], run[i + 1])

        idx = int(np.searchsorted(e.traj.t, self.sim_t))
        if idx > 5:
            trail(e.traj.r[max(0, idx - 20):idx:2], COL_THREAT)
        if self.interceptor_path is not None:
            ip, it = self.interceptor_path
            j = min(int(np.searchsorted(it, self.sim_t)), len(ip) - 1)
            if j > 5:
                trail(ip[max(0, j - 20):j:2], COL_INTER)

    def _draw_intercept_math(self, s, view, e, sol, int_pos, int_vel,
                             threat_pos, threat_vel):
        """The intercept calculation, evaluated at the current instant.

        Every line is computed here from live state rather than recalled from
        the solution dict, so what is displayed is the arithmetic actually
        being performed, not a caption describing it.
        """
        pg = self.pg
        panel = pg.Rect(view.right - 430, view.y + 60, 420, 366)
        self._panel(s, panel, "INTERCEPT MATH  (live)")
        x = panel.x + 12
        y = [panel.y + 34]

        def line(txt, col=COL_TEXT):
            self._text(s, txt, (x, y[0]), self.font_s, col)
            y[0] += 16

        def head(txt):
            y[0] += 6
            self._text(s, txt, (x, y[0]), self.font_s, COL_ACCENT)
            y[0] += 17

        head("1. FIRE CONTROL -- Lambert boundary value problem")
        line("   given r1 (site), r2 (PIP), time of flight t:")
        line("   solve for v1 such that the conic joins r1 -> r2 in t")
        r1 = float(np.linalg.norm(e.site_r)) / 1000.0
        r2 = float(np.linalg.norm(sol["r_pip"])) / 1000.0
        line(f"   |r1| = {r1:,.0f} km   |r2| = {r2:,.0f} km   "
             f"t = {sol['flight_time_s']:.0f} s", COL_DIM)
        line(f"   |v1| required     = {np.linalg.norm(sol['v_required'])/1000:.3f} km/s",
             COL_PIP)
        line(f"   platform velocity = {np.linalg.norm(e.v_site)/1000:.3f} km/s",
             COL_DIM)
        line(f"   dv bought = |v1 - v_site| = {sol['dv_required']/1000:.3f} km/s",
             COL_PIP)
        line(f"   budget {e.dv_available/1000:.3f} km/s  ->  margin "
             f"{sol['margin_ms']:+,.0f} m/s",
             COL_INTER if sol["margin_ms"] > 0 else COL_WARN)

        head("2. CLOSING GEOMETRY")
        if int_pos is not None:
            rel = int_pos - threat_pos
            vrel = int_vel - threat_vel
            rmag = float(np.linalg.norm(rel))
            vmag = float(np.linalg.norm(vrel))
            t_go = (-float(np.dot(rel, vrel)) / (vmag * vmag)
                    if vmag > 1e-9 else float("nan"))
            line(f"   r_rel = {rmag:,.1f} m      v_rel = {vmag/1000:.3f} km/s")
            line("   t_go  = -(r_rel . v_rel)/|v_rel|^2")
            line(f"         = {t_go:,.3f} s", COL_PIP)
            zem = rel + vrel * t_go if t_go == t_go else rel
            u = vrel / max(vmag, 1e-9)
            zem_p = zem - float(np.dot(zem, u)) * u
            line("   ZEM   = r_rel + v_rel t_go    (predicted miss vector)")
            line(f"   |ZEM_perp| = {float(np.linalg.norm(zem_p)):,.2f} m", COL_PIP)
            n_gain = e.spec["kv_nav_gain"]
            if t_go == t_go and t_go > 1e-3:
                a_cmd = n_gain * float(np.linalg.norm(zem_p)) / (t_go * t_go)
                line(f"   a_cmd = N' |ZEM_perp| / t_go^2, N'={n_gain:.0f}")
                line(f"         = {a_cmd:,.2f} m/s2   "
                     f"(limit {e.spec['kv_divert_accel_ms2']:.0f})",
                     COL_WARN if a_cmd > e.spec["kv_divert_accel_ms2"] else COL_PIP)

        head("3. PERTURBATION -- differential gravity")
        if int_pos is not None:
            g_i = gravity(int_pos)
            g_t = gravity(threat_pos)
            dg = float(np.linalg.norm(g_i - g_t))
            line(f"   g(KV) - g(target) = {dg:.3e} m/s2")
            line(f"   free drift over 1 s = {0.5*dg:.4f} m", COL_DIM)
            line("   PN absorbs this; it shows up as divert, not miss.",
                 COL_DIM)

        head("4. OUTCOME (this run)")
        lr = e.spec["kv_lethal_radius_m"]
        line(f"   miss {self.term_miss_m:,.3f} m   vs lethal radius {lr:.2f} m",
             COL_INTER if self.term_hit else COL_WARN)
        line(f"   divert used {self.term_dv_used:,.1f} of "
             f"{e.spec['kv_divert_dv_ms']:.0f} m/s",
             COL_WARN if self.term_dv_used >= e.spec["kv_divert_dv_ms"] * 0.99
             else COL_TEXT)

    def _draw_collision(self, s, view, cam, contact_pos):
        """Expanding kinetic-impact flash at the real contact point. Kinetic
        only -- hit-to-kill carries no explosive, and nothing here should
        read as one; the burst is a visibility aid, not a warhead effect."""
        pg = self.pg
        q = cam.project(contact_pos, view)
        if not q:
            return
        cx, cy = int(q[0]), int(q[1])
        dt = self.sim_t - self.term_cpa_t
        t_frac = min(1.0, dt / 2.0)
        base_r = int(20 + t_frac * 200)
        for layer in range(6):
            r = int(base_r * (1.0 - layer * 0.12))
            if r <= 0:
                continue
            alpha = max(0, 255 - layer * 40 - int(t_frac * 100))
            if layer < 2:
                col = (255, min(255, 200 + alpha // 3), 60)
            elif layer < 4:
                col = (alpha, alpha // 2, 30)
            else:
                col = (alpha // 2, alpha // 4, 20)
            col = tuple(min(255, c) for c in col)
            pg.draw.circle(s, col, (cx, cy), r)
            pg.gfxdraw.aacircle(s, cx, cy, r, col)
        n_spokes = 16
        spoke_len = int(30 + t_frac * 300)
        for i in range(n_spokes):
            ang = 2 * math.pi * i / n_spokes
            ex = cx + int(spoke_len * math.cos(ang))
            ey = cy + int(spoke_len * math.sin(ang))
            fade = max(0, 255 - int(t_frac * 200))
            col = (fade, max(0, fade // 3), 20)
            pg.draw.aaline(s, col, (cx, cy), (ex, ey))
        if dt < 0.3:
            flash_r = int(40 + dt * 200)
            pg.draw.circle(s, (255, 255, 240), (cx, cy), flash_r, 2)

    def _draw_flyby(self, s, view, cam, int_pos, threat_pos):
        """Measured clearance at closest approach -- both objects frozen at
        their real simulated endpoints, joined by a labelled scale line."""
        pg = self.pg
        t_proj = cam.project(threat_pos, view)
        i_proj = cam.project(int_pos, view)
        if t_proj:
            self._glow_marker(s, COL_THREAT, int(t_proj[0]), int(t_proj[1]), 6)
            self._text(s, "threat", (int(t_proj[0]) + 12, int(t_proj[1]) - 6),
                       self.font_s, COL_THREAT)
        if i_proj:
            self._glow_marker(s, COL_INTER, int(i_proj[0]), int(i_proj[1]), 6)
            self._text(s, "kill vehicle",
                       (int(i_proj[0]) + 12, int(i_proj[1]) - 6),
                       self.font_s, COL_INTER)
        if t_proj and i_proj:
            pg.draw.aaline(s, COL_WARN, (int(i_proj[0]), int(i_proj[1])),
                           (int(t_proj[0]), int(t_proj[1])))
            mid_x = (int(i_proj[0]) + int(t_proj[0])) // 2
            mid_y = (int(i_proj[1]) + int(t_proj[1])) // 2
            self._text(s, f"{self.term_miss_m:.2f} m", (mid_x + 8, mid_y - 6),
                       self.font_s, COL_WARN)

    # --- MODE 2: 3D VEHICLE ---

    def _mode_interceptor(self, s, rect):
        pg = self.pg
        spec = INTERCEPTOR
        if self._mesh is None:
            self._mesh = build_interceptor_mesh(spec)
        cam = self.cam_v
        view = pg.Rect(rect.x, rect.y, rect.w - 430, rect.h)
        side = pg.Rect(rect.right - 422, rect.y + 8, 414, rect.h - 16)

        light = np.array([0.42, 0.66, 0.62])
        light /= np.linalg.norm(light)
        Rc = cam.matrix()
        polys = []
        labels = []

        for p in self._mesh:
            v = p["verts"].copy()
            if self.explode:
                v = v + np.array([0.0, 0.0, p["group"] * 2600.0])
            # camera space
            q = (v - cam.center) @ Rc.T
            q[:, 2] += cam.dist
            for f in p["faces"]:
                pts = q[list(f)]
                if np.any(pts[:, 2] <= 1e3):
                    continue
                wpts = v[list(f)]
                cen = wpts.mean(axis=0)
                if self.section and cen[0] > 0.0:
                    continue                       # cutaway: drop the near half
                n = np.cross(wpts[1] - wpts[0], wpts[2] - wpts[0])
                ln = np.linalg.norm(n)
                if ln < 1e-9:
                    continue
                n = n / ln
                diff = max(0.0, abs(float(np.dot(n, light))))
                # Two-tone shading: ambient + diffuse + rim light
                sh = 0.22 + 0.58 * diff + 0.20 * (1.0 - diff) ** 2
                col = tuple(min(255, int(c * sh)) for c in p["color"])
                # Edge colour: slightly darker than fill for definition
                edge = tuple(max(0, int(c * 0.55)) for c in col)
                focal = min(view.w, view.h) * 1.05
                scr = [(view.centerx + focal * a[0] / a[2],
                        view.centery - focal * a[1] / a[2]) for a in pts]
                polys.append((float(pts[:, 2].mean()), scr, col, edge))
            if p["name"] and self.labels:
                c = v.mean(axis=0)
                cq = (c - cam.center) @ Rc.T
                cq[2] += cam.dist
                if cq[2] > 1e3:
                    focal = min(view.w, view.h) * 1.05
                    labels.append((view.centerx + focal * cq[0] / cq[2],
                                   view.centery - focal * cq[1] / cq[2],
                                   p["name"], p["color"]))

        polys.sort(key=lambda t: -t[0])            # painter's algorithm
        for _, scr, col, edge in polys:
            pg.draw.polygon(s, col, scr)
            try:
                pg.gfxdraw.aapolygon(s, [(int(x), int(y)) for x, y in scr],
                                     edge)
            except Exception:
                pass

        for lx, ly, name, col in labels:
            self._aacircle(s, col, int(lx), int(ly), 3)
            pg.draw.aaline(s, (90, 104, 128), (lx, ly), (lx + 26, ly - 20))
            self._text(s, name, (lx + 30, ly - 30), self.font_s,
                       (200, 214, 234))

        self._text(s, f"{spec['name']}   {spec['length_mm']/1000:.1f} m x "
                      f"{spec['diameter_mm']/1000:.2f} m   -- true scale, "
                      f"{len(polys)} faces",
                   (view.x + 24, view.y + 14), self.font_b, COL_ACCENT)
        self._text(s, "drag orbit   wheel zoom   E exploded   X section cut   "
                      "L labels", (view.x + 24, view.y + 40), self.font_s,
                   COL_PIP)
        st = []
        if self.explode:
            st.append(("EXPLODED", COL_WARN))
        if self.section:
            st.append(("SECTION CUT", COL_PIP))
        if st:
            bx = view.x + 24
            by = view.y + 62
            for label, col in st:
                tw = self.font_s.size(label)[0] + 12
                pg.draw.rect(s, (28, 36, 52), pg.Rect(bx, by, tw, 18),
                             border_radius=3)
                pg.draw.rect(s, col, pg.Rect(bx, by, tw, 18), 1,
                             border_radius=3)
                self._text(s, label, (bx + 6, by + 2), self.font_s, col)
                bx += tw + 6

        # Scale ruler at the bottom of the 3D view
        ruler_y = view.bottom - 30
        ruler_len_m = spec["length_mm"] / 1000.0
        focal = min(view.w, view.h) * 1.05
        # Project a 1-metre segment at the vehicle centre depth to get px/m
        cz = cam.dist
        px_per_m = focal / cz
        ruler_px = int(ruler_len_m * px_per_m)
        if ruler_px > 20:
            rx = view.centerx - ruler_px // 2
            pg.draw.line(s, COL_DIM, (rx, ruler_y),
                         (rx + ruler_px, ruler_y), 1)
            pg.draw.line(s, COL_DIM, (rx, ruler_y - 4),
                         (rx, ruler_y + 4), 1)
            pg.draw.line(s, COL_DIM, (rx + ruler_px, ruler_y - 4),
                         (rx + ruler_px, ruler_y + 4), 1)
            self._text(s, f"{ruler_len_m:.1f} m",
                       (rx + ruler_px + 8, ruler_y - 6), self.font_s, COL_DIM)

        # --- spec panel ---
        self._panel(s, side, "SPECIFICATION")
        y = [side.y + 42]

        def row(k, v, src=""):
            self._text(s, k, (side.x + 14, y[0]), self.font_s, COL_DIM)
            self._text(s, v, (side.x + 210, y[0]), self.font_s, COL_TEXT)
            if src:
                self._text(s, src, (side.x + 330, y[0]), self.font_s,
                           COL_WARN if src == "est." else COL_DIM)
            y[0] += 20

        def head(t):
            y[0] += 10
            self._text(s, t, (side.x + 14, y[0]), self.font, COL_ACCENT)
            y[0] += 22

        row("overall length", f"{spec['length_mm']/1000:.2f} m", "published")
        row("diameter", f"{spec['diameter_mm']/1000:.2f} m", "published")
        # Visual stage breakdown bar
        y[0] += 6
        total_len = spec["length_mm"]
        stage_cols = [(52, 68, 94), (62, 82, 112), (74, 98, 132),
                      (104, 132, 168), (206, 168, 74)]
        stage_labels = ["S1", "S2", "S3", "Shroud", "KV"]
        stage_lens = [spec["stage1_len_mm"], spec["stage2_len_mm"],
                      spec["stage3_len_mm"], spec["shroud_len_mm"],
                      spec["kv_len_mm"]]
        bar_x = side.x + 14
        bar_w = side.w - 28
        bar_h = 16
        bx_offset = 0
        for i, (sl, sc_col, sl_lbl) in enumerate(zip(stage_lens, stage_cols,
                                                      stage_labels)):
            sw = max(2, int(bar_w * sl / total_len))
            pg.draw.rect(s, sc_col, pg.Rect(bar_x + bx_offset, y[0], sw, bar_h))
            hl = tuple(min(255, c + 25) for c in sc_col)
            pg.draw.line(s, hl, (bar_x + bx_offset, y[0]),
                         (bar_x + bx_offset + sw, y[0]), 1)
            if sw > 20:
                self._text(s, sl_lbl, (bar_x + bx_offset + sw // 2 - 6,
                           y[0] + 2), self.font_s, (180, 190, 210))
            bx_offset += sw
        pg.draw.rect(s, COL_GRID, pg.Rect(bar_x, y[0], bar_w, bar_h), 1)
        y[0] += bar_h + 4
        head("ENERGY")
        row("burnout velocity", f"{spec['burnout_v_ms']/1000:.2f} km/s", "est.")
        # Energy margin bar: burnout v vs circular orbital speed
        v_orb = math.sqrt(MU / R_E)
        v_burn = spec["burnout_v_ms"]
        em_x = side.x + 14
        em_y = y[0]
        em_w = side.w - 28
        em_h = 10
        pg.draw.rect(s, COL_DARK, pg.Rect(em_x, em_y, em_w, em_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(em_x, em_y, em_w, em_h), 1)
        orb_frac = min(1.0, v_orb / v_burn)
        orb_px = int(orb_frac * em_w)
        pg.draw.line(s, COL_PIP, (em_x + orb_px, em_y - 2),
                     (em_x + orb_px, em_y + em_h + 2), 2)
        pg.draw.aaline(s, COL_PIP, (em_x + orb_px, em_y - 2),
                       (em_x + orb_px, em_y + em_h + 2))
        burn_px = int(min(1.0, v_burn / v_orb) * em_w)
        pg.draw.rect(s, COL_INTER, pg.Rect(em_x, em_y, burn_px, em_h))
        hl = tuple(min(255, c + 30) for c in COL_INTER)
        pg.draw.line(s, hl, (em_x, em_y), (em_x + burn_px, em_y), 1)
        self._text(s, f"orbital speed {v_orb/1000:.2f} km/s (yellow line)",
                   (em_x, em_y + em_h + 4), self.font_s, COL_DIM)
        y[0] += em_h + 22
        row("burn time", f"{spec['burn_time_s']:.0f} s", "est.")
        row("decision delay", f"{spec['launch_delay_s']:.0f} s", "est.")
        head("KILL VEHICLE")
        row("mass", f"{spec['kv_mass_kg']:.0f} kg", "published")
        row("divert budget", f"{spec['kv_divert_dv_ms']:.0f} m/s", "est.")
        row("divert accel", f"{spec['kv_divert_accel_ms2']:.0f} m/s2", "est.")
        row("seeker acquisition", f"{spec['kv_seeker_acq_km']:.0f} km", "est.")
        row("seeker noise", f"{spec['kv_seeker_noise_urad']:.0f} urad", "est.")
        row("guidance lag", f"{spec['kv_guidance_lag_s']:.2f} s", "est.")
        row("aimpoint sigma", f"{spec['kv_aimpoint_sigma_m']:.2f} m", "est.")
        row("nav constant N'", f"{spec['kv_nav_gain']:.1f}", "textbook")
        row("lethal radius", f"{spec['kv_lethal_radius_m']:.2f} m", "contact")
        y[0] += 14
        for ln in ("No warhead appears in this model because the",
                   "system has none. At the closing speeds in",
                   "mode 3 the collision is the whole mechanism,",
                   "which is why the lethal radius is half a",
                   "metre rather than tens of metres.",
                   "",
                   "Values marked est. are estimates and drive",
                   "Pk more than anything else here."):
            self._text(s, ln, (side.x + 14, y[0]), self.font_s, COL_DIM)
            y[0] += 18

    # --- MODE 3 ---

    def _mode_geometry(self, s, rect):
        pg = self.pg
        e = self.eng
        if not e.solution or not e.pk:
            self._text(s, "no feasible intercept", (rect.x + 30, rect.y + 30),
                       self.font_b, COL_WARN)
            return
        sol = e.solution
        left = pg.Rect(rect.x + 16, rect.y + 12, rect.w // 2 - 24, rect.h - 30)
        right = pg.Rect(rect.centerx + 8, rect.y + 12, rect.w // 2 - 24,
                        rect.h - 30)
        self._panel(s, left, "INTERCEPT TRIANGLE")
        self._panel(s, right, "MISS DISTANCE DISTRIBUTION")

        cx, cy = left.centerx, left.y + 150
        vt, vi = sol["v_pip"], sol["v_arrival"]
        sc = 95.0 / max(float(np.linalg.norm(vt)), float(np.linalg.norm(vi)))

        def p2(v):
            return (cx + v[1] * sc, cy - v[0] * sc)

        # Velocity vectors with glow underlayer and arrowheads
        self._glow_line(s, [(cx, cy), p2(vt)], COL_THREAT, 3)
        self._glow_line(s, [(cx, cy), p2(vi)], COL_INTER, 3)
        self._arrowhead(s, COL_THREAT, (cx, cy), p2(vt), 12)
        self._arrowhead(s, COL_INTER, (cx, cy), p2(vi), 12)
        pg.draw.aaline(s, COL_PIP, p2(vt), p2(vi))
        # Origin marker
        self._aacircle(s, COL_ACCENT, int(cx), int(cy), 4)
        self._text(s, f"threat {np.linalg.norm(vt)/1000:.2f} km/s",
                   (p2(vt)[0] + 8, p2(vt)[1]), self.font_s, COL_THREAT)
        self._text(s, f"interceptor {np.linalg.norm(vi)/1000:.2f} km/s",
                   (p2(vi)[0] + 8, p2(vi)[1]), self.font_s, COL_INTER)
        # Angle arc between velocity vectors
        ang_t = math.atan2(-(p2(vt)[1] - cy), p2(vt)[0] - cx)
        ang_i = math.atan2(-(p2(vi)[1] - cy), p2(vi)[0] - cx)
        arc_r = 28
        # Determine sweep direction
        if abs(ang_t - ang_i) < math.pi:
            a0, a1 = min(ang_t, ang_i), max(ang_t, ang_i)
        else:
            a0, a1 = max(ang_t, ang_i), min(ang_t, ang_i) + 2 * math.pi
        n_arc = max(6, int(abs(a1 - a0) * 20))
        arc_pts = []
        for ia in range(n_arc + 1):
            aa = a0 + (a1 - a0) * ia / n_arc
            arc_pts.append((cx + arc_r * math.cos(aa),
                           cy - arc_r * math.sin(aa)))
        if len(arc_pts) > 1:
            pg.draw.aalines(s, COL_DIM, False, arc_pts)
        # Angle label
        ang_deg = abs(math.degrees(ang_t - ang_i))
        if ang_deg > 180:
            ang_deg = 360 - ang_deg
        mid_a = (a0 + a1) / 2
        self._text(s, f"{ang_deg:.0f} deg",
                   (cx + (arc_r + 8) * math.cos(mid_a) - 18,
                    cy - (arc_r + 8) * math.sin(mid_a) - 6),
                   self.font_s, COL_DIM)
        self._text(s, f"closing {sol['closing_speed']/1000:.2f} km/s",
                   (cx - 70, cy + 130), self.font_b, COL_PIP)
        # Closing speed resultant vector (dashed line showing the sum)
        v_close = vt - vi
        p_close = p2(v_close)
        # Draw resultant as a dim dashed-style line
        n_dash = 12
        for id in range(n_dash):
            t0 = id / n_dash
            t1 = (id + 0.5) / n_dash
            p0 = (cx + (p_close[0] - cx) * t0, cy + (p_close[1] - cy) * t0)
            p1 = (cx + (p_close[0] - cx) * t1, cy + (p_close[1] - cy) * t1)
            pg.draw.aaline(s, COL_PIP, p0, p1)
        self._arrowhead(s, COL_PIP, (cx, cy), p_close, 8)
        self._text(s, f"resultant {np.linalg.norm(v_close)/1000:.2f} km/s",
                   (p_close[0] + 8, p_close[1] + 4), self.font_s, COL_PIP)
        # Kinetic energy gauge below the closing speed text
        ke = 0.5 * INTERCEPTOR["kv_mass_kg"] * sol["closing_speed"] ** 2
        ke_gauge_x = left.x + 24
        ke_gauge_y = cy + 160
        ke_gauge_w = left.w - 48
        ke_gauge_h = 8
        pg.draw.rect(s, COL_DARK, pg.Rect(ke_gauge_x, ke_gauge_y, ke_gauge_w, ke_gauge_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(ke_gauge_x, ke_gauge_y, ke_gauge_w, ke_gauge_h), 1)
        ke_tnt_g = ke / 4.184e9
        ke_fill = int(min(1.0, ke_tnt_g / 5.0) * ke_gauge_w)
        if ke_fill > 0:
            ke_col = COL_PIP if ke_tnt_g > 1.0 else COL_WARN
            pg.draw.rect(s, ke_col, pg.Rect(ke_gauge_x, ke_gauge_y, ke_fill, ke_gauge_h))
            hl = tuple(min(255, c + 30) for c in ke_col)
            pg.draw.line(s, hl, (ke_gauge_x, ke_gauge_y),
                         (ke_gauge_x + ke_fill, ke_gauge_y), 1)
        self._text(s, f"KE gauge: {ke_tnt_g:.2f} t TNT (bar = 5 t max)",
                   (ke_gauge_x, ke_gauge_y + ke_gauge_h + 2), self.font_s, COL_DIM)

        y = left.y + 330
        t_hom = INTERCEPTOR["kv_seeker_acq_km"] * 1000 / sol["closing_speed"]
        for k, v in (("closing speed", f"{sol['closing_speed']/1000:.2f} km/s"),
                     ("KV mass", f"{INTERCEPTOR['kv_mass_kg']:.0f} kg"),
                     ("impact energy", f"{ke/4.184e9:.2f} t TNT equivalent"),
                     ("", ""),
                     ("homing time from acq", f"{t_hom:.1f} s"),
                     ("track error at handover", f"{e.track_sigma:,.0f} m"),
                     ("divert available", f"{INTERCEPTOR['kv_divert_dv_ms']:.0f} m/s"),
                     ("divert used, median", f"{e.pk['dv_median']:.0f} m/s"),
                     ("divert used, p95", f"{e.pk['dv_p95']:.0f} m/s")):
            if k:
                self._text(s, k, (left.x + 24, y), self.font_s, COL_DIM)
                self._text(s, v, (left.x + 270, y), self.font_s, COL_TEXT)
            y += 21
        # Impact energy bar (visual reference: 1 t TNT = 4.184 GJ)
        ke_tnt = ke / 4.184e9
        ie_x, ie_y = left.x + 24, y
        ie_w, ie_h = left.w - 48, 10
        pg.draw.rect(s, COL_DARK, pg.Rect(ie_x, ie_y, ie_w, ie_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(ie_x, ie_y, ie_w, ie_h), 1)
        # Scale: 5 t TNT full bar
        ie_fill = int(min(1.0, ke_tnt / 5.0) * ie_w)
        ie_col = COL_THREAT if ke_tnt > 2.0 else COL_WARN if ke_tnt > 1.0 else COL_INTER
        pg.draw.rect(s, ie_col, pg.Rect(ie_x, ie_y, ie_fill, ie_h))
        hl = tuple(min(255, c + 30) for c in ie_col)
        pg.draw.line(s, hl, (ie_x, ie_y), (ie_x + ie_fill, ie_y), 1)
        # 1 t TNT reference line
        ref_x = ie_x + int(ie_w * 0.2)
        pg.draw.line(s, COL_PIP, (ref_x, ie_y - 2), (ref_x, ie_y + ie_h + 2), 1)
        pg.draw.aaline(s, COL_PIP, (ref_x, ie_y - 2), (ref_x, ie_y + ie_h + 2))
        self._text(s, f"impact energy: {ke_tnt:.2f} t TNT  (yellow = 1 t)",
                   (ie_x, ie_y + ie_h + 4), self.font_s, COL_DIM)
        y += ie_h + 22
        # Divert budget bar: shows how much of the divert budget is consumed
        dv_avail = INTERCEPTOR["kv_divert_dv_ms"]
        dv_med = e.pk["dv_median"]
        dv_p95 = e.pk["dv_p95"]
        db_x, db_y = left.x + 24, y + 4
        db_w, db_h = left.w - 48, 12
        pg.draw.rect(s, COL_DARK, pg.Rect(db_x, db_y, db_w, db_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(db_x, db_y, db_w, db_h), 1)
        # Median divert fill
        med_fill = int(min(1.0, dv_med / dv_avail) * db_w)
        if med_fill > 0:
            pg.draw.rect(s, COL_INTER, pg.Rect(db_x, db_y, med_fill, db_h))
        # p95 marker
        p95_x = db_x + int(min(1.0, dv_p95 / dv_avail) * db_w)
        pg.draw.line(s, COL_WARN, (p95_x, db_y - 2), (p95_x, db_y + db_h + 2), 2)
        pg.draw.aaline(s, COL_WARN, (p95_x, db_y - 2), (p95_x, db_y + db_h + 2))
        self._text(s, "divert budget: median (green) vs p95 (orange line)",
                   (db_x, db_y + db_h + 6), self.font_s, COL_DIM)

        misses = e.pk["misses"]
        hx, hy = right.x + 46, right.y + 72
        hw, hh = right.w - 92, 250
        pg.draw.rect(s, COL_DARK, pg.Rect(hx, hy, hw, hh))
        pg.draw.rect(s, COL_GRID, pg.Rect(hx, hy, hw, hh), 1)
        # Horizontal grid lines for count reference
        for g in range(1, 5):
            gy = hy + hh * g / 5
            pg.draw.line(s, (18, 24, 38), (hx, int(gy)), (hx + hw, int(gy)), 1)
        # Lethal zone background shading (drawn before bars)
        lr = INTERCEPTOR["kv_lethal_radius_m"]
        top = max(1.0, float(np.percentile(misses, 98)))
        lx_pre = int(hx + hw * min(1.0, lr / top))
        lethal_bg = tuple(int(c * 0.12) for c in COL_INTER)
        pg.draw.rect(s, lethal_bg, pg.Rect(hx, hy, max(1, lx_pre - hx), hh))
        bins = 40
        hist, edges = np.histogram(np.clip(misses, 0, top), bins=bins,
                                   range=(0, top))
        mx = max(1, int(hist.max()))
        for i, c in enumerate(hist):
            bh = hh * c / mx
            bx = hx + i * hw / bins
            lethal = edges[i] <= INTERCEPTOR["kv_lethal_radius_m"]
            bar_col = COL_INTER if lethal else COL_THREAT
            # Slight top highlight on bars
            pg.draw.rect(s, bar_col,
                         pg.Rect(int(bx), int(hy + hh - bh),
                                 max(1, int(hw / bins) - 1), int(bh)))
            if bh > 3:
                hl = tuple(min(255, v + 30) for v in bar_col)
                pg.draw.line(s, hl,
                             (int(bx), int(hy + hh - bh)),
                             (int(bx) + max(1, int(hw / bins) - 1),
                              int(hy + hh - bh)), 1)
        lr = INTERCEPTOR["kv_lethal_radius_m"]
        lx = hx + hw * min(1.0, lr / top)
        pg.draw.line(s, COL_PIP, (lx, hy), (lx, hy + hh), 2)
        pg.draw.aaline(s, COL_PIP, (lx, hy), (lx, hy + hh))
        self._text(s, f"lethal radius {lr:.1f} m", (lx + 6, hy + 6),
                   self.font_s, COL_PIP)
        # Median miss marker
        med_x = hx + hw * min(1.0, e.pk["miss_median"] / top)
        pg.draw.line(s, COL_ACCENT, (med_x, hy), (med_x, hy + hh), 1)
        pg.draw.aaline(s, COL_ACCENT, (med_x, hy), (med_x, hy + hh))
        self._text(s, f"median {e.pk['miss_median']:.2f} m",
                   (med_x + 4, hy + hh - 16), self.font_s, COL_ACCENT)
        self._text(s, "0 m", (hx, hy + hh + 6), self.font_s, COL_DIM)
        self._text(s, f"{top:,.0f} m", (hx + hw - 60, hy + hh + 6),
                   self.font_s, COL_DIM)

        y = hy + hh + 42
        for k, v, col in (("median miss", f"{e.pk['miss_median']:.2f} m", COL_TEXT),
                          ("95th percentile", f"{e.pk['miss_p95']:.2f} m", COL_TEXT),
                          ("single-shot Pk", f"{e.pk['pk']*100:.1f}%",
                           COL_INTER if e.pk["pk"] > 0.7 else COL_WARN)):
            self._text(s, k, (right.x + 34, y), self.font_s, COL_DIM)
            self._text(s, v, (right.x + 290, y), self.font, col)
            y += 24
        y += 12
        for ln in ("Green bars are hits. This distribution is driven almost",
                   "entirely by track error at handover, not by the guidance",
                   "law -- raise the track error and the whole histogram walks",
                   "right, past the lethal radius, no matter how good the",
                   "kill vehicle is."):
            self._text(s, ln, (right.x + 34, y), self.font_s, COL_DIM)
            y += 19

    # --- MODE 4 ---

    def _mode_sensors(self, s, rect):
        pg = self.pg
        e = self.eng
        y = rect.y + 30
        self._text(s, "RADAR RANGE EQUATION", (rect.x + 40, y), self.font_b,
                   COL_ACCENT)
        y += 32
        self._text(s, "R_max = [ Pt G^2 lambda^2 sigma n / "
                      "( (4pi)^3 k T B F (S/N) ) ] ^ 1/4",
                   (rect.x + 60, y), self.font, COL_TEXT)
        y += 40

        for key, band in (("ew_radar", "UHF"), ("xband_radar", "X")):
            sen = SENSORS[key]
            rmax = radar_max_range(sen, RCS_M2[band])
            self._text(s, sen["name"], (rect.x + 40, y), self.font_b,
                       COL_SENSOR)
            y += 26
            for k, v in (("frequency", f"{sen['freq_hz']/1e9:.3f} GHz"),
                         ("peak power", f"{sen['power_w']/1e3:,.0f} kW"),
                         ("antenna gain", f"{sen['gain_dbi']:.0f} dBi"),
                         ("required S/N", f"{sen['snr_req_db']:.0f} dB"),
                         ("pulses integrated", f"{sen['integration_n']:,}"),
                         ("assumed RCS", f"{RCS_M2[band]:.3f} m2"),
                         ("-> detection range", f"{rmax/1000:,.0f} km")):
                self._text(s, "    " + k, (rect.x + 40, y), self.font_s,
                           COL_DIM)
                self._text(s, v, (rect.x + 280, y), self.font_s,
                           COL_PIP if k.startswith("->") else COL_TEXT)
                y += 19
            # Radar range comparison bar
            rb_x = rect.x + 40
            rb_y = y + 2
            rb_w = 300
            rb_h = 8
            rmax_all = max(radar_max_range(SENSORS["ew_radar"], RCS_M2["UHF"]),
                           radar_max_range(SENSORS["xband_radar"], RCS_M2["X"]))
            rb_col = COL_SENSOR if key == "xband_radar" else (86, 120, 190)
            pg.draw.rect(s, COL_DARK, pg.Rect(rb_x, rb_y, rb_w, rb_h))
            pg.draw.rect(s, COL_GRID, pg.Rect(rb_x, rb_y, rb_w, rb_h), 1)
            rb_fill = int(min(1.0, rmax / rmax_all) * rb_w)
            if rb_fill > 0:
                pg.draw.rect(s, rb_col, pg.Rect(rb_x, rb_y, rb_fill, rb_h))
                hl = tuple(min(255, c + 30) for c in rb_col)
                pg.draw.line(s, hl, (rb_x, rb_y), (rb_x + rb_fill, rb_y), 1)
            y += 16
            y += 14

        gx, gcy, gr = rect.centerx + 240, rect.y + 300, 190
        # Earth disc with limb shading and atmospheric glow
        for ggr, gga in ((gr + 10, 20), (gr + 6, 40), (gr + 3, 70)):
            glow = tuple(min(255, int(c * gga / 255)) for c in COL_ATMOS)
            pg.draw.circle(s, glow, (gx, gcy), ggr, 1)
        for i in range(gr, 0, -2):
            t = 1.0 - i / gr
            shade = int(35 * t)
            col = (min(255, COL_EARTH[0] + shade),
                   min(255, COL_EARTH[1] + shade),
                   min(255, COL_EARTH[2] + shade))
            pg.draw.circle(s, col, (gx, gcy), i)
        pg.gfxdraw.aacircle(s, gx, gcy, gr, COL_LIMB)
        pg.draw.circle(s, COL_LIMB, (gx, gcy), int(gr * 1.02), 1)
        self._text(s, "HORIZON LIMIT", (gx - 70, rect.y + 70), self.font_b,
                   COL_ACCENT)
        for alt_km, col in ((200, (70, 100, 150)), (600, (100, 140, 200)),
                            (1200, COL_SENSOR)):
            h = radar_horizon_m(alt_km * 1000.0, 100.0)
            ang = min(1.15, h / (R_E * 2.4) * 1.1)
            px = gx + gr * math.sin(ang) * 1.30
            py = gcy - gr * math.cos(ang) * 1.30
            pg.draw.aaline(s, col, (gx, gcy - gr), (px, py))
            pg.draw.line(s, col, (gx, gcy - gr), (px, py), 2)
            self._text(s, f"{alt_km} km alt -> {h/1000:,.0f} km",
                       (px + 8, py - 8), self.font_s, col)

        y2 = rect.bottom - 150
        self._text(s, "IN THIS ENGAGEMENT", (rect.x + 40, y2), self.font_b,
                   COL_ACCENT)
        y2 += 28
        det = "never" if e.t_detect is None else f"T+{e.t_detect/60:.1f} min"
        icp = ("n/a" if not e.solution
               else f"T+{e.solution['t_intercept']/60:.1f} min")
        for k, v in (("first detection", det),
                     ("launch decision delay", f"{INTERCEPTOR['launch_delay_s']:.0f} s"),
                     ("interceptor boost", f"{INTERCEPTOR['burn_time_s']:.0f} s"),
                     ("intercept", icp)):
            self._text(s, "    " + k, (rect.x + 40, y2), self.font_s, COL_DIM)
            self._text(s, v, (rect.x + 300, y2), self.font_s, COL_TEXT)
            y2 += 20
        # Detection-to-intercept chain bar
        if e.t_detect is not None and e.solution:
            bt = e.meta["boost_time_s"]
            total = e.meta["total_time_s"]
            chain_y = y2 + 6
            chain_x = rect.x + 40
            chain_w = 360
            def cx(t):
                return chain_x + chain_w * min(1.0, t / total)
            d = e.t_detect + bt
            b0 = d + INTERCEPTOR["launch_delay_s"]
            b1 = b0 + INTERCEPTOR["burn_time_s"]
            ic = e.solution["t_intercept"] + bt
            # Blind phase
            pg.draw.rect(s, (58, 58, 70), pg.Rect(int(cx(0)), chain_y,
                         max(2, int(cx(d) - cx(0))), 10))
            # Decision delay
            pg.draw.rect(s, COL_WARN, pg.Rect(int(cx(d)), chain_y,
                         max(2, int(cx(b0) - cx(d))), 10))
            # Boost
            pg.draw.rect(s, (60, 140, 100), pg.Rect(int(cx(b0)), chain_y,
                         max(2, int(cx(b1) - cx(b0))), 10))
            # Coast
            pg.draw.rect(s, COL_INTER, pg.Rect(int(cx(b1)), chain_y,
                         max(2, int(cx(ic) - cx(b1))), 10))
            pg.draw.rect(s, COL_GRID, pg.Rect(chain_x, chain_y, chain_w, 10), 1)
            # Intercept marker
            pg.draw.line(s, COL_PIP, (int(cx(ic)), chain_y - 2),
                         (int(cx(ic)), chain_y + 12), 2)
            pg.draw.aaline(s, COL_PIP, (int(cx(ic)), chain_y - 2),
                         (int(cx(ic)), chain_y + 12))
            self._text(s, "detect", (int(cx(d)) - 14, chain_y + 14),
                       self.font_s, COL_DIM)
            self._text(s, "intercept", (int(cx(ic)) - 24, chain_y + 14),
                       self.font_s, COL_PIP)
            y2 += 30
        self._text(s, "A radar cannot see through the Earth, and for most "
                      "engagements the horizon binds long before the range",
                   (rect.x + 40, y2 + 10), self.font_s, COL_DIM)
        self._text(s, "equation does. That is why detection TIME, not radar "
                      "power, is what sets how much battlespace exists.",
                   (rect.x + 40, y2 + 29), self.font_s, COL_DIM)

    # --- MODE 5 ---

    def _mode_timeline(self, s, rect):
        pg = self.pg
        e = self.eng
        if not e.solution:
            self._text(s, "no feasible intercept", (rect.x + 30, rect.y + 30),
                       self.font_b, COL_WARN)
            return
        sol = e.solution
        total = e.meta["total_time_s"]
        x0, x1 = rect.x + 240, rect.right - 90
        span = x1 - x0
        bt = e.meta["boost_time_s"]

        def tx(t):
            return x0 + span * min(1.0, max(0.0, t / total))

        self._text(s, "Every bar is time the defence does not get back. "
                      "Scale is the full threat flight.",
                   (rect.x + 44, rect.y + 36), self.font, COL_DIM)
        # ICBM flight phase bands (above the bars)
        band_y = rect.y + 60
        band_h = 16
        # Boost phase
        boost_x = int(tx(0))
        boost_end_x = int(tx(bt))
        pg.draw.rect(s, (120, 60, 56), pg.Rect(boost_x, band_y,
                     max(2, boost_end_x - boost_x), band_h))
        hl = tuple(min(255, c + 30) for c in (120, 60, 56))
        pg.draw.line(s, hl, (boost_x, band_y), (boost_end_x, band_y), 1)
        self._text(s, "boost", (boost_x + 4, band_y + 1), self.font_s,
                   (200, 160, 150))
        # Midcourse phase
        mid_x = boost_end_x
        mid_end_x = int(tx(bt + e.meta["tof_s"]))
        pg.draw.rect(s, (90, 46, 44), pg.Rect(mid_x, band_y,
                     max(2, mid_end_x - mid_x), band_h))
        hl = tuple(min(255, c + 30) for c in (90, 46, 44))
        pg.draw.line(s, hl, (mid_x, band_y), (mid_end_x, band_y), 1)
        self._text(s, "midcourse", (mid_x + 4, band_y + 1), self.font_s,
                   (200, 160, 150))
        # Reentry phase (from apogee to impact)
        re_x = mid_end_x
        re_end_x = int(tx(total))
        if re_end_x > re_x:
            pg.draw.rect(s, (60, 36, 34), pg.Rect(re_x, band_y,
                         max(2, re_end_x - re_x), band_h))
            hl = tuple(min(255, c + 30) for c in (60, 36, 34))
            pg.draw.line(s, hl, (re_x, band_y), (re_end_x, band_y), 1)
            self._text(s, "reentry", (re_x + 4, band_y + 1), self.font_s,
                       (180, 140, 130))
        pg.draw.rect(s, COL_GRID, pg.Rect(int(tx(0)), band_y,
                     max(2, int(tx(total)) - int(tx(0))), band_h), 1)
        y = rect.y + 84
        bars = [("threat boost", 0.0, bt, (120, 60, 56)),
                ("threat midcourse", bt, bt + e.meta["tof_s"], (90, 46, 44))]
        if e.t_detect is not None:
            d = e.t_detect + bt
            b0 = d + INTERCEPTOR["launch_delay_s"]
            b1 = b0 + INTERCEPTOR["burn_time_s"]
            bars += [("blind (no track)", 0.0, d, (58, 58, 70)),
                     ("decision delay", d, b0, COL_WARN),
                     ("interceptor boost", b0, b1, (60, 140, 100)),
                     ("interceptor coast", b1, sol["t_intercept"] + bt,
                      COL_INTER)]

        # Phase boundary reference lines (span all bars vertically)
        phase_y0 = rect.y + 84
        phase_y1 = rect.y + 84 + len(bars) * 36 + 12
        for t_phase, label in [(bt, "boost end"), (bt + e.meta["tof_s"], "apogee")]:
            px_phase = int(tx(t_phase))
            pg.draw.line(s, (40, 52, 72), (px_phase, phase_y0),
                         (px_phase, phase_y1), 1)
            pg.draw.aaline(s, (40, 52, 72), (px_phase, phase_y0),
                           (px_phase, phase_y1))

        for name, a, b, col in bars:
            xa, xb = tx(a), tx(b)
            bw = max(2, int(xb - xa))
            pg.draw.rect(s, col, pg.Rect(int(xa), int(y), bw, 26))
            # Subtle top highlight for depth
            hl = tuple(min(255, c + 35) for c in col)
            pg.draw.line(s, hl, (int(xa), int(y)), (int(xa) + bw, int(y)), 1)
            # Dark bottom edge
            dk = tuple(max(0, c - 30) for c in col)
            pg.draw.line(s, dk, (int(xa), int(y) + 26),
                         (int(xa) + bw, int(y) + 26), 1)
            self._text(s, name, (rect.x + 44, y + 5), self.font_s, COL_TEXT)
            self._text(s, f"{b - a:.0f} s", (xb + 8, y + 5), self.font_s,
                       COL_DIM)
            y += 36

        ipx = tx(sol["t_intercept"] + bt)
        pg.draw.line(s, COL_PIP, (ipx, rect.y + 76), (ipx, y + 12), 2)
        pg.draw.aaline(s, COL_PIP, (ipx, rect.y + 76), (ipx, y + 12))
        self._text(s, "INTERCEPT", (ipx - 42, y + 18), self.font_s, COL_PIP)
        if e.t_last:
            lx = tx(e.t_last + bt)
            pg.draw.line(s, COL_WARN, (lx, rect.y + 76), (lx, y + 12), 2)
            pg.draw.aaline(s, COL_WARN, (lx, rect.y + 76), (lx, y + 12))
            self._text(s, "window closes", (lx - 44, y + 38), self.font_s,
                       COL_WARN)

        # Time scale ruler at the bottom of the chart
        ruler_y = y + 12
        for t_mark in range(0, int(total) + 1, 120):
            rx = int(tx(t_mark))
            pg.draw.line(s, COL_GRID, (rx, ruler_y), (rx, ruler_y + 4), 1)
            if t_mark % 300 == 0:
                self._text(s, f"{t_mark/60:.0f}m", (rx - 8, ruler_y + 6),
                           self.font_s, COL_DIM)
        pg.draw.line(s, COL_GRID, (x0, ruler_y), (x1, ruler_y), 1)

        y += 82
        self._text(s, "MARGIN ANALYSIS", (rect.x + 44, y), self.font_b,
                   COL_ACCENT)
        y += 32
        cycle = (INTERCEPTOR["burn_time_s"] + INTERCEPTOR["launch_delay_s"] + 30)
        for k, v in (("battlespace open for", f"{e.battlespace_s:,.0f} s"),
                     ("shoot-look-shoot cycle", f"{cycle:.0f} s"),
                     ("shot opportunities", f"{e.shot_opportunities}"),
                     ("intercept altitude", f"{sol['alt_km']:,.0f} km"),
                     ("exoatmospheric floor", f"{EXO_FLOOR_M/1000:.0f} km")):
            self._text(s, k, (rect.x + 64, y), self.font_s, COL_DIM)
            self._text(s, v, (rect.x + 360, y), self.font_s, COL_TEXT)
            y += 22
        # Shoot-look-shoot cycle visual: how many cycles fit in battlespace
        if e.battlespace_s > 0 and cycle > 0:
            cy_y = y + 6
            cy_x = rect.x + 64
            cy_w = min(400, rect.w - 120)
            cy_h = 14
            # Battlespace background
            pg.draw.rect(s, COL_DARK, pg.Rect(cy_x, cy_y, cy_w, cy_h))
            pg.draw.rect(s, COL_GRID, pg.Rect(cy_x, cy_y, cy_w, cy_h), 1)
            # Cycle segments
            n_cycles = int(e.battlespace_s / cycle)
            seg_w = int(cy_w * cycle / e.battlespace_s)
            for ic in range(min(n_cycles, 10) + 1):
                sx = cy_x + ic * seg_w
                if ic < n_cycles:
                    seg_col = COL_INTER if ic < e.shot_opportunities else (40, 80, 60)
                    pg.draw.rect(s, seg_col, pg.Rect(sx, cy_y, min(seg_w, cy_x + cy_w - sx), cy_h))
                    hl = tuple(min(255, c + 25) for c in seg_col)
                    pg.draw.line(s, hl, (sx, cy_y), (sx + min(seg_w, cy_x + cy_w - sx), cy_y), 1)
                # Cycle boundary
                pg.draw.line(s, COL_DIM, (sx, cy_y - 2), (sx, cy_y + cy_h + 2), 1)
            self._text(s, f"battlespace {e.battlespace_s:.0f}s / cycle {cycle:.0f}s = {n_cycles} cycles",
                       (cy_x, cy_y + cy_h + 4), self.font_s, COL_DIM)
            y += cy_h + 24
        self._text(s, "The shoot-look-shoot cycle decides whether a second "
                      "attempt exists at all. If it exceeds the",
                   (rect.x + 44, y + 16), self.font_s, COL_DIM)
        self._text(s, "battlespace, the defence gets exactly one attempt "
                      "regardless of how many interceptors it owns.",
                   (rect.x + 44, y + 35), self.font_s, COL_DIM)

    # --- MODE 6 ---

    def _mode_battlespace(self, s, rect):
        pg = self.pg
        if self._bs_cache is None:
            self._text(s, "computing engagements across the range band...",
                       (rect.x + 44, rect.y + 60), self.font_b, COL_PIP)
            self.pg.display.flip()
            self._bs_cache = [(rk, Engagement(range_km=rk, trials=60))
                              for rk in (1000, 2500, 4500, 7000, 10000, 13000)]
        y = rect.y + 40
        self._text(s, "WINDOW AND SHOT COUNT VS THREAT RANGE",
                   (rect.x + 44, y), self.font_b, COL_ACCENT)
        y += 42
        self._text(s, "  RANGE      APOGEE    FLIGHT   INTERCEPT   WINDOW  "
                      "SHOTS     Pk", (rect.x + 44, y), self.font, COL_DIM)
        y += 26
        bar_x = rect.x + 600
        bar_w = 120
        for rk, e in self._bs_cache:
            if not e.solution or not e.pk:
                self._text(s, f"{rk:>7,} km {e.meta['apogee_km']:>8,.0f} km"
                              f" {e.meta['total_time_s']/60:>6.1f} m      no shot",
                           (rect.x + 44, y), self.font, COL_WARN)
            else:
                pk = e.pk["pk"]
                self._text(s, f"{rk:>7,} km {e.meta['apogee_km']:>8,.0f} km"
                              f" {e.meta['total_time_s']/60:>6.1f} m"
                              f" {e.solution['alt_km']:>8,.0f} km"
                              f" {e.battlespace_s:>7.0f} s"
                              f" {e.shot_opportunities:>5}"
                              f" {pk*100:>5.0f}%",
                           (rect.x + 44, y), self.font, COL_TEXT)
                # Inline Pk bar
                pk_col = COL_INTER if pk > 0.5 else COL_WARN if pk > 0.1 else COL_THREAT
                pg.draw.rect(s, COL_DARK, pg.Rect(bar_x, y + 2, bar_w, 12))
                fill = int(pk * bar_w)
                if fill > 0:
                    pg.draw.rect(s, pk_col, pg.Rect(bar_x, y + 2, fill, 12))
                    hl = tuple(min(255, c + 30) for c in pk_col)
                    pg.draw.line(s, hl, (bar_x, y + 2), (bar_x + fill, y + 2), 1)
                pg.draw.rect(s, COL_GRID, pg.Rect(bar_x, y + 2, bar_w, 12), 1)
                # Inline battlespace window bar (below Pk bar)
                win_max = 600.0
                win_col = COL_INTER if e.battlespace_s > 120 else COL_WARN if e.battlespace_s > 30 else COL_THREAT
                pg.draw.rect(s, COL_DARK, pg.Rect(bar_x, y + 16, bar_w, 4))
                win_fill = int(min(1.0, e.battlespace_s / win_max) * bar_w)
                if win_fill > 0:
                    pg.draw.rect(s, win_col, pg.Rect(bar_x, y + 16, win_fill, 4))
            y += 24
        y += 30
        for ln in ("Longer-range threats give a midcourse interceptor MORE",
                   "OPPORTUNITY, which reads backwards until you look at the",
                   "apogee column: a longer arc spends more time high and slow,",
                   "so the window widens and extra shots fit.",
                   "",
                   "Read the Pk column against it -- the trends point opposite",
                   "ways. Per-shot Pk drifts DOWN as range grows, because a",
                   "longer arc also means a longer coast between the last sensor",
                   "update and handover, and track error grows with that coast.",
                   "Longer range buys opportunities and spends track quality.",
                   "",
                   "The short-range rows are where midcourse defence runs out of",
                   "geometry -- the arc never gets high enough or lasts long",
                   "enough. That regime is what terminal-phase systems exist to",
                   "cover, and it is why no single layer suffices."):
            self._text(s, ln, (rect.x + 44, y), self.font, COL_DIM)
            y += 22

    # --- MODE 7 ---

    def _mode_discrimination(self, s, rect):
        pg = self.pg
        e = self.eng
        if not e.pk:
            self._text(s, "no feasible intercept", (rect.x + 30, rect.y + 30),
                       self.font_b, COL_WARN)
            return
        p = e.pk["pk"]
        if p <= 0.0:
            self._text(s, "Pk is zero at this track quality; nothing to dilute.",
                       (rect.x + 44, rect.y + 40), self.font_b, COL_WARN)
            return
        y = rect.y + 36
        self._text(s, "OBJECT COUNT AGAINST A FIXED INTERCEPTOR INVENTORY",
                   (rect.x + 44, y), self.font_b, COL_ACCENT)
        y += 34
        self._text(s, f"Single-shot Pk against one object at this engagement's "
                      f"track quality: {p*100:.0f}%",
                   (rect.x + 44, y), self.font, COL_TEXT)
        # Pk gauge bar
        gk_x = rect.x + 44
        gk_y = y + 22
        gk_w = min(300, rect.w // 3)
        gk_h = 8
        pg.draw.rect(s, COL_DARK, pg.Rect(gk_x, gk_y, gk_w, gk_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(gk_x, gk_y, gk_w, gk_h), 1)
        gk_fill = int(p * gk_w)
        gk_col = COL_INTER if p > 0.5 else COL_WARN if p > 0.1 else COL_THREAT
        if gk_fill > 0:
            pg.draw.rect(s, gk_col, pg.Rect(gk_x, gk_y, gk_fill, gk_h))
            hl = tuple(min(255, c + 30) for c in gk_col)
            pg.draw.line(s, hl, (gk_x, gk_y), (gk_x + gk_fill, gk_y), 1)
        ref_x = gk_x + gk_w // 2
        pg.draw.line(s, COL_DIM, (ref_x, gk_y - 2), (ref_x, gk_y + gk_h + 2), 1)
        self._text(s, f"{p*100:.0f}%", (gk_x + gk_w + 8, gk_y - 2),
                   self.font_s, gk_col)
        y += 40

        cx, cy = rect.x + 64, y
        cw, ch = rect.w // 2 - 130, 280
        pg.draw.rect(s, COL_DARK, pg.Rect(cx, cy, cw, ch))
        pg.draw.rect(s, COL_GRID, pg.Rect(cx, cy, cw, ch), 1)
        # Grid lines for reference
        for g in range(1, 5):
            gy = cy + ch * g / 5
            pg.draw.line(s, (18, 24, 38), (cx, int(gy)), (cx + cw, int(gy)), 1)
        for g in range(1, 5):
            gx = cx + cw * g / 5
            pg.draw.line(s, (18, 24, 38), (int(gx), cy), (int(gx), cy + ch), 1)
        base = math.log(0.01) / math.log(max(1e-9, 1.0 - p))
        ns = list(range(1, 21))
        needs = [base * n for n in ns]
        mx = max(needs)
        pts = [(cx + cw * (n - 1) / (len(ns) - 1),
                cy + ch - ch * needs[i] / mx) for i, n in enumerate(ns)]
        # Filled area under the curve for visual weight
        fill_pts = pts + [(cx + cw, cy + ch), (cx, cy + ch)]
        fill_col = tuple(max(0, int(c * 0.15)) for c in COL_WARN)
        pg.draw.polygon(s, fill_col, [(int(px), int(py)) for px, py in fill_pts])
        # Glow underlayer on the curve
        glow = tuple(max(0, int(c * 0.35)) for c in COL_WARN)
        pg.draw.lines(s, glow, False, pts, 5)
        pg.draw.lines(s, COL_WARN, False, pts, 3)
        pg.draw.aalines(s, COL_WARN, False, pts)
        self._text(s, "interceptors needed for 99% intercept",
                   (cx + 12, cy + 10), self.font_s, COL_WARN)
        self._text(s, "1 object", (cx, cy + ch + 8), self.font_s, COL_DIM)
        self._text(s, "20 objects", (cx + cw - 74, cy + ch + 8), self.font_s,
                   COL_DIM)
        self._text(s, f"{mx:.0f}", (cx - 36, cy), self.font_s, COL_DIM)

        tx, ty = rect.centerx + 70, y
        self._text(s, "OBJECTS   FOR 99%   LEAKAGE @ 4 COMMITTED",
                   (tx, ty), self.font, COL_DIM)
        ty += 28
        for n in (1, 2, 4, 6, 10, 15, 20):
            need = math.ceil(base * n)
            leak = (1.0 - p) ** (4.0 / n)
            col = COL_INTER if leak < 0.05 else (COL_PIP if leak < 0.3 else COL_WARN)
            self._text(s, f"{n:>7}   {need:>7}   {leak*100:>19.1f}%",
                       (tx, ty), self.font, col)
            # Inline leakage bar
            lb_x = tx + 260
            lb_w = 80
            pg.draw.rect(s, COL_DARK, pg.Rect(lb_x, ty + 2, lb_w, 12))
            lb_fill = int(min(1.0, leak) * lb_w)
            if lb_fill > 0:
                pg.draw.rect(s, col, pg.Rect(lb_x, ty + 2, lb_fill, 12))
                hl = tuple(min(255, c + 30) for c in col)
                pg.draw.line(s, hl, (lb_x, ty + 2), (lb_x + lb_fill, ty + 2), 1)
            pg.draw.rect(s, COL_GRID, pg.Rect(lb_x, ty + 2, lb_w, 12), 1)
            ty += 24

        y = cy + ch + 46
        for ln in ("The left curve grows linearly and the right column collapses.",
                   "That is the entire argument, and it makes no claim about how",
                   "hard the objects are to tell apart -- it is what happens when",
                   "they are NOT told apart. In vacuum there is no drag to sort",
                   "them by mass, so every object stays a candidate.",
                   "",
                   "This is stated as an inventory question on purpose. How many",
                   "interceptors a defence needs is the decision-relevant",
                   "direction and the one a physics model should help with. The",
                   "inverse question is not in this file and should not be added.",
                   "",
                   "It is also why midcourse defence is described as effective",
                   "against a small, unsophisticated or accidental launch and not",
                   "a deliberate one. Same hardware, different claim -- and the",
                   "whole difference is this table."):
            self._text(s, ln, (rect.x + 44, y), self.font, COL_DIM)
            y += 21

    # --- MODE 10: GUN SANDBOX ---

    def _bar(self, s, x, y, w, h, frac, ok):
        """Horizontal margin bar. Green under the limit, red over it, with the
        limit marked -- so 'how far past' is visible, not just pass/fail."""
        pg = self.pg
        pg.draw.rect(s, COL_DARK, pg.Rect(x, y, w, h))
        pg.draw.rect(s, COL_GRID, pg.Rect(x, y, w, h), 1)
        # The limit sits at 60% of the bar width, so a value up to ~1.67x the
        # limit still has bar left to render and the overshoot stays visible.
        full_scale = 1.0 / 0.6
        limx = x + int(w * 0.6)
        fill = int(min(frac / full_scale, 1.0) * w)
        bar_col = COL_INTER if ok else COL_THREAT
        pg.draw.rect(s, bar_col,
                     pg.Rect(x + 1, y + 1, max(1, fill - 1), h - 2))
        # Top highlight on the fill for depth
        if fill > 2:
            hl = tuple(min(255, c + 30) for c in bar_col)
            pg.draw.line(s, hl, (x + 1, y + 1), (x + fill, y + 1), 1)
        pg.draw.line(s, COL_PIP, (limx, y - 3), (limx, y + h + 3), 2)
        pg.draw.aaline(s, COL_PIP, (limx, y - 3), (limx, y + h + 3))
        return limx

    def _mode_gun(self, s, rect):
        pg = self.pg
        mat = MATERIAL_NAMES[self.gun_mat]
        g = gun_assessment(self.gun_v, mat, self.gun_barrel, self.gun_nose)
        mach = self.gun_v / 343.0

        self._text(s, "Drive the parameters yourself. Every number below is "
                      "computed live from the equations in PHYSICS.",
                   (rect.x + 40, rect.y + 26), self.font, COL_DIM)
        self._text(s, "M material    - / =  muzzle velocity    "
                      "[ / ]  barrel length    N  nose radius",
                   (rect.x + 40, rect.y + 48), self.font, COL_PIP)
        # Stated where a reader would go looking for the missing drawing.
        self._text(s, "NO GUN OR PROJECTILE IS DRAWN HERE, DELIBERATELY -- "
                      "margins and pass/fail only.",
                   (rect.x + 40, rect.y + 68), self.font_s, COL_WARN)
        self._text(s, "A critique tells you whether it closes; a model tells "
                      "you how to build one. See ABOUT.",
                   (rect.x + 40, rect.y + 84), self.font_s, COL_DIM)

        # --- parameter block ---
        y = rect.y + 112
        self._text(s, "INPUTS", (rect.x + 40, y), self.font_b, COL_ACCENT)
        y += 30
        for k, v in (("material", mat),
                     ("muzzle velocity", f"{self.gun_v:,.0f} m/s   Mach {mach:.1f}"),
                     ("barrel length", f"{self.gun_barrel:.1f} m"),
                     ("nose radius", f"{self.gun_nose*1000:.0f} mm"),
                     ("projectile mass", f"{g['mass_kg']:.2f} kg")):
            self._text(s, k, (rect.x + 56, y), self.font_s, COL_DIM)
            self._text(s, v, (rect.x + 230, y), self.font_s, COL_TEXT)
            y += 21
        self._text(s, MATERIALS[mat]["note"], (rect.x + 56, y + 4),
                   self.font_s, COL_SENSOR)

        # --- four checks ---
        bx, by, bw = rect.x + 40, y + 50, min(560, rect.w // 2 - 80)
        self._text(s, "WHAT PHYSICS DOES TO THEM", (bx, by), self.font_b,
                   COL_ACCENT)
        by += 34
        checks = [
            ("1  BARREL STRESS",
             f"{g['sigma_req']/1e9:.2f} GPa needed / {g['sigma_limit']/1e9:.1f} GPa available",
             g["stress_ratio"], g["pass_barrel"],
             f"survivable to Mach {g['v_max_barrel']/343:.1f} in this barrel;"
             f" Mach {mach:.0f} needs {g['barrel_required']:.0f} m"),
            ("2  ATMOSPHERIC EXIT",
             f"keeps {g['v_kept']*100:.0f}% -> Mach {g['v_after_atmos']/343:.1f} on the way out",
             1.0 - g["v_kept"], g["v_kept"] > 0.5,
             f"ballistic coefficient {g['beta']:,.0f} kg/m2"),
            ("3  STAGNATION HEATING",
             f"{g['t_wall']:,.0f} K wall / {g['t_limit']:,.0f} K limit",
             g["thermal_ratio"], g["pass_thermal"],
             f"{g['q_wm2']/1e6:,.0f} MW/m2 = {g['q_wm2']/1e6/5:,.0f}x Apollo peak"),
            ("4  ENERGY PER SHOT",
             f"{g['input_j']/1e6:,.0f} MJ electrical in",
             g["input_j"] / 1e9, g["input_j"] < 1e9,
             f"{g['ke_j']/4.184e9:.2f} t TNT of muzzle energy"),
        ]
        for name, val, ratio, ok, note in checks:
            # Pass/fail icon
            icon_col = COL_INTER if ok else COL_THREAT
            icon_x = bx - 16
            icon_y = by + 4
            if ok:
                # Checkmark
                pg.draw.aaline(s, icon_col, (icon_x, icon_y + 4),
                               (icon_x + 4, icon_y + 8))
                pg.draw.aaline(s, icon_col, (icon_x + 4, icon_y + 8),
                               (icon_x + 10, icon_y))
            else:
                # X mark
                pg.draw.aaline(s, icon_col, (icon_x, icon_y),
                               (icon_x + 8, icon_y + 8))
                pg.draw.aaline(s, icon_col, (icon_x + 8, icon_y),
                               (icon_x, icon_y + 8))
            self._text(s, name, (bx, by), self.font, COL_TEXT)
            self._text(s, val, (bx + 220, by), self.font_s,
                       COL_INTER if ok else COL_THREAT)
            self._bar(s, bx, by + 22, bw, 14, ratio, ok)
            self._text(s, note, (bx, by + 40), self.font_s, COL_DIM)
            by += 74

        # --- verdict panel ---
        px = rect.x + rect.w // 2 + 40
        py = rect.y + 112
        panel = pg.Rect(px - 16, py - 16, min(520, rect.w // 2 - 60), 300)
        self._panel(s, panel, "VERDICT")
        py += 26
        fails = [n for n, _, _, ok, _ in checks if not ok]
        if not fails:
            self._text(s, "All four checks pass at this setting.",
                       (px, py), self.font, COL_INTER)
            py += 26
            self._text(s, "Note what that does NOT establish: no launcher has",
                       (px, py), self.font_s, COL_DIM); py += 19
            self._text(s, "reached this velocity for a mass like this. The",
                       (px, py), self.font_s, COL_DIM); py += 19
            self._text(s, "binding constraint there is armature physics, not",
                       (px, py), self.font_s, COL_DIM); py += 19
            self._text(s, "the projectile -- see the record bars below.",
                       (px, py), self.font_s, COL_DIM); py += 19
        else:
            self._text(s, f"FAILS {len(fails)} of 4 checks:", (px, py),
                       self.font, COL_THREAT)
            py += 24
            for f in fails:
                self._text(s, "   " + f, (px, py), self.font_s, COL_THREAT)
                py += 19
            py += 8
            self._text(s, "Red bars run past the yellow limit line. Raising",
                       (px, py), self.font_s, COL_DIM); py += 19
            self._text(s, "muzzle velocity moves checks 3 and 4 the wrong way",
                       (px, py), self.font_s, COL_DIM); py += 19
            self._text(s, "faster than a longer barrel fixes check 1.",
                       (px, py), self.font_s, COL_DIM); py += 19

        # --- demonstrated-velocity reference bars ---
        ry = panel.bottom + 40
        self._text(s, "WHAT LAUNCHERS HAVE ACTUALLY ACHIEVED (kg-scale)",
                   (px, ry), self.font_b, COL_ACCENT)
        ry += 30
        refs = [("Navy railgun, demonstrated", 2500.0, COL_INTER),
                ("Light-gas gun, record", 11000.0, COL_PIP),
                ("your setting", self.gun_v, COL_SENSOR),
                ("Mach 62", 21266.0, COL_THREAT)]
        scale = min(430, rect.w // 2 - 120) / 22000.0
        for name, v, col in refs:
            w = max(2, int(v * scale))
            pg.draw.rect(s, col, pg.Rect(px, ry, w, 15))
            # Top highlight
            hl = tuple(min(255, c + 35) for c in col)
            pg.draw.line(s, hl, (px, ry), (px + w, ry), 1)
            # Anti-aliased right edge
            pg.draw.aaline(s, col, (px + w, ry), (px + w, ry + 15))
            self._text(s, f"{name}  {v:,.0f} m/s  (Mach {v/343:.0f})",
                       (px, ry + 18), self.font_s, COL_DIM)
            ry += 46

    # --- MODE 11: LAYERED DEFENCE ---

    def _mode_layered(self, s, rect):
        pg = self.pg
        e = self.eng
        if not e.pk or e.pk["pk"] <= 0.0:
            self._text(s, "no feasible intercept at this threat range",
                       (rect.x + 40, rect.y + 40), self.font_b, COL_WARN)
            return
        p = e.pk["pk"]
        k, n = self.lay_layers, self.lay_objects

        self._text(s, "[ / ]  independent sites        O / P  objects the "
                      "defence cannot separate",
                   (rect.x + 40, rect.y + 30), self.font, COL_PIP)
        self._text(s, f"Single-chain Pk at this engagement's track quality: "
                      f"{p*100:.0f}%",
                   (rect.x + 40, rect.y + 54), self.font, COL_TEXT)
        # Pk gauge bar
        gk_x = rect.x + 40
        gk_y = rect.y + 72
        gk_w = min(300, rect.w // 3)
        gk_h = 8
        pg.draw.rect(s, COL_DARK, pg.Rect(gk_x, gk_y, gk_w, gk_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(gk_x, gk_y, gk_w, gk_h), 1)
        gk_fill = int(p * gk_w)
        gk_col = COL_INTER if p > 0.5 else COL_WARN if p > 0.1 else COL_THREAT
        if gk_fill > 0:
            pg.draw.rect(s, gk_col, pg.Rect(gk_x, gk_y, gk_fill, gk_h))
            hl = tuple(min(255, c + 30) for c in gk_col)
            pg.draw.line(s, hl, (gk_x, gk_y), (gk_x + gk_fill, gk_y), 1)
        # 50% reference line
        ref_x = gk_x + gk_w // 2
        pg.draw.line(s, COL_DIM, (ref_x, gk_y - 2), (ref_x, gk_y + gk_h + 2), 1)
        self._text(s, f"{p*100:.0f}%", (gk_x + gk_w + 8, gk_y - 2),
                   self.font_s, gk_col)

        leak_layer = (1.0 - p) ** k
        pc, _ = salvo_probability(p, k, 0.9, trials=6000)
        leak_salvo = 1.0 - pc
        leak_real = (1.0 - p) ** (k / n)

        y = rect.y + 96
        self._text(s, f"SPENDING {k} INTERCEPTORS TWO WAYS  (against 1 object)",
                   (rect.x + 40, y), self.font_b, COL_ACCENT)
        y += 34
        for lbl, leak, col in (
                (f"{k} rounds, ONE battery (shared track)", leak_salvo, COL_THREAT),
                (f"{k} INDEPENDENT sites (separate tracks)", leak_layer, COL_INTER)):
            self._text(s, lbl, (rect.x + 56, y), self.font_s, COL_DIM)
            self._text(s, f"leak {_pct(leak)}%", (rect.x + 400, y), self.font_s,
                       col)
            bw = int(min(1.0, leak / 0.06) * 300)
            pg.draw.rect(s, col, pg.Rect(rect.x + 560, y + 2, max(2, bw), 13))
            # Top highlight
            if bw > 2:
                hl = tuple(min(255, c + 35) for c in col)
                pg.draw.line(s, hl, (rect.x + 560, y + 2),
                             (rect.x + 560 + bw, y + 2), 1)
            y += 30
        self._text(s, "Same interceptor count. The only difference is whether "
                      "the errors are independent.",
                   (rect.x + 56, y + 6), self.font_s, COL_DIM)
        # Visual comparison bar: salvo vs independent
        cmp_x = rect.x + 56
        cmp_y = y + 28
        cmp_w = min(600, rect.w - 120)
        cmp_h = 10
        pg.draw.rect(s, COL_DARK, pg.Rect(cmp_x, cmp_y, cmp_w, cmp_h))
        pg.draw.rect(s, COL_GRID, pg.Rect(cmp_x, cmp_y, cmp_w, cmp_h), 1)
        # Salvo leak (red, left)
        salvo_fill = int(min(1.0, leak_salvo / 0.5) * cmp_w)
        if salvo_fill > 0:
            pg.draw.rect(s, COL_THREAT, pg.Rect(cmp_x, cmp_y, salvo_fill, cmp_h))
            hl = tuple(min(255, c + 30) for c in COL_THREAT)
            pg.draw.line(s, hl, (cmp_x, cmp_y), (cmp_x + salvo_fill, cmp_y), 1)
        # Independent leak (green, overlaid from left)
        layer_fill = int(min(1.0, leak_layer / 0.5) * cmp_w)
        if layer_fill > 0:
            pg.draw.rect(s, COL_INTER, pg.Rect(cmp_x, cmp_y + cmp_h + 2,
                         layer_fill, cmp_h))
            hl = tuple(min(255, c + 30) for c in COL_INTER)
            pg.draw.line(s, hl, (cmp_x, cmp_y + cmp_h + 2),
                         (cmp_x + layer_fill, cmp_y + cmp_h + 2), 1)
            pg.draw.rect(s, COL_GRID, pg.Rect(cmp_x, cmp_y + cmp_h + 2,
                         cmp_w, cmp_h), 1)
        self._text(s, f"salvo leak {_pct(leak_salvo)}%  vs  independent leak {_pct(leak_layer)}%",
                   (cmp_x, cmp_y + 2 * cmp_h + 8), self.font_s, COL_DIM)

        # --- grid ---
        gy = y + 54
        self._text(s, "LEAKAGE  --  rows = independent layers, cols = objects "
                      "on track", (rect.x + 40, gy), self.font_b, COL_ACCENT)
        gy += 32
        cols = [1, 2, 4, 6, 10, 15, 20, 30]
        cw, ch = 82, 26
        self._text(s, "layers", (rect.x + 44, gy), self.font_s, COL_DIM)
        for j, nn in enumerate(cols):
            self._text(s, f"{nn:>4}", (rect.x + 120 + j * cw, gy), self.font_s,
                       COL_PIP if nn == n else COL_DIM)
        gy += 22
        for i in range(1, 9):
            self._text(s, f"{i:>4}", (rect.x + 52, gy + 4), self.font_s,
                       COL_PIP if i == k else COL_DIM)
            for j, nn in enumerate(cols):
                lk = (1.0 - p) ** (i / nn)
                # green = held, red = leaking
                col = (int(60 + 180 * min(1.0, lk * 2.2)),
                       int(200 - 150 * min(1.0, lk * 2.2)), 90)
                r = pg.Rect(rect.x + 116 + j * cw, gy, cw - 6, ch - 4)
                pg.draw.rect(s, col, r, border_radius=3)
                # Subtle inner highlight
                hl = tuple(min(255, c + 20) for c in col)
                pg.draw.line(s, hl, (r.x + 2, r.y + 1),
                             (r.right - 2, r.y + 1), 1)
                if i == k and nn == n:
                    pg.draw.rect(s, COL_ACCENT, r, 2, border_radius=3)
                self._text(s, f"{lk*100:5.1f}%", (r.x + 8, r.y + 4),
                           self.font_s, (12, 16, 24))
            gy += ch
        gy += 16
        self._text(s, f"selected: {k} layers vs {n} objects   ->   "
                      f"leakage {_pct(leak_real)}%   "
                      f"intercept {_pct(1-leak_real)}%",
                   (rect.x + 44, gy), self.font, COL_ACCENT)
        gy += 30
        for ln in ("Down a column leakage falls geometrically -- independent "
                   "chains compound.",
                   "Along a row it decays only as k/n, because every layer "
                   "faces the same ambiguity.",
                   "Independence is purchasable. Ambiguity is not, and that is "
                   "what caps the defence."):
            self._text(s, ln, (rect.x + 44, gy), self.font_s, COL_DIM)
            gy += 20

    # --- MODE 12 ---

    def _mode_physics(self, s, rect):
        pg = self.pg
        e = self.eng
        m, sol = e.meta, e.solution
        L = [("CONIC RANGE EQUATION  (sizes the threat arc from range alone)", COL_ACCENT),
             ("  psi   = range / R_E            gamma = pi/4 - psi/4", COL_TEXT),
             ("  Q = v^2 r / mu                 e = sqrt(1 + Q(Q-2) cos^2 gamma)", COL_TEXT),
             ("  p = (r v cos gamma)^2 / mu     r = p / (1 + e cos nu)", COL_TEXT),
             (f"  -> gamma {m['gamma_deg']:.1f} deg,  v {m['v_burnout_ms']/1000:.3f} km/s,"
              f"  apogee {m['apogee_km']:,.0f} km,  free flight {m['tof_s']/60:.1f} min", COL_PIP),
             ("  Limit check: psi -> pi gives gamma -> 0 and v -> sqrt(mu/R_E)", COL_DIM),
             ("  = 7.90 km/s, i.e. circular orbital speed. Correct antipodal answer.", COL_DIM),
             ("", COL_TEXT),
             ("EQUATIONS OF MOTION  (RK4, J2, drag below 200 km)", COL_ACCENT),
             ("  a = -mu r/|r|^3  +  J2 term  -  0.5 rho |v_rel| v_rel / beta", COL_TEXT),
             (f"  J2 = {J2:.6e}    beta = {THREAT_BETA:,.0f} kg/m^2", COL_DIM),
             ("", COL_TEXT),
             ("LAMBERT SOLUTION  (the fire-control problem)", COL_ACCENT),
             ("  Given r1 (interceptor site) and r2 (predicted intercept point)", COL_TEXT),
             ("  and time of flight, solve for the burnout velocity required.", COL_TEXT),
             ("  y(z) = r1 + r2 + A (z S(z) - 1)/sqrt(C(z))", COL_TEXT),
             ("  t(z) = ( x^3 S(z) + A sqrt(y) )/sqrt(mu),   x = sqrt(y/C)", COL_TEXT)]
        if sol:
            L.append((f"  -> dv required {sol['dv_required']/1000:.3f} km/s of "
                      f"{INTERCEPTOR['burnout_v_ms']/1000:.3f} available "
                      f"(margin {sol['margin_ms']/1000:+.3f})", COL_PIP))
        L += [("", COL_TEXT),
              ("PROPORTIONAL NAVIGATION  (zero-effort-miss form)", COL_ACCENT),
              ("  t_go  = -(r_rel . v_rel) / |v_rel|^2", COL_TEXT),
              ("  ZEM   = r_rel + v_rel t_go", COL_TEXT),
              (f"  a_cmd = N' ZEM_perp / t_go^2        N' = {INTERCEPTOR['kv_nav_gain']:.1f}", COL_TEXT),
              ("  Exoatmospheric: no aerodynamic control exists, so every command", COL_DIM),
              ("  is spent from a finite divert budget and cannot be recovered.", COL_DIM)]
        if sol and e.pk:
            L += [(f"  -> divert median {e.pk['dv_median']:.0f} m/s, p95 {e.pk['dv_p95']:.0f} m/s"
                   f" of {INTERCEPTOR['kv_divert_dv_ms']:.0f} available", COL_PIP),
                  (f"  -> median miss {e.pk['miss_median']:.2f} m,  Pk {e.pk['pk']*100:.0f}%", COL_PIP)]
        L += [("", COL_TEXT),
              ("RADAR RANGE EQUATION", COL_ACCENT),
              ("  R^4 = Pt G^2 lambda^2 sigma n / ( (4pi)^3 k T B F (S/N) )", COL_TEXT),
              (f"  -> UHF {radar_max_range(SENSORS['ew_radar'], RCS_M2['UHF'])/1000:,.0f} km"
               f"   X-band {radar_max_range(SENSORS['xband_radar'], RCS_M2['X'])/1000:,.0f} km", COL_PIP),
              ("", COL_TEXT),
              ("WHAT IS NOT COMPUTED HERE", COL_ACCENT),
              ("  Threat vehicle design, propulsion, staging, reentry-vehicle or", COL_DIM),
              ("  warhead physics, and countermeasure design. The arc above needs", COL_DIM),
              ("  none of it -- gravity sets the trajectory, and that is precisely", COL_DIM),
              ("  why the intercept is a solvable kinematics problem.", COL_DIM)]
        y = rect.y + 30 - self.scroll * 8
        for txt, col in L:
            if rect.y <= y <= rect.bottom - 10:
                # Section headers get a subtle divider line
                if col == COL_ACCENT and txt:
                    div_w = self.font.size(txt)[0]
                    pg.draw.line(s, (28, 38, 56), (rect.x + 48, y + 18),
                                 (rect.x + 48 + div_w + 8, y + 18), 1)
                self._text(s, txt, (rect.x + 48, y), self.font, col)
            y += 22

    # --- MODE 9 ---

    def _mode_about(self, s, rect):
        pg = self.pg
        L = [("ICBMI -- BALLISTIC MISSILE DEFENCE INTERCEPT DIGITAL TWIN", COL_ACCENT),
             ("", COL_TEXT),
             ("WHAT THIS IS", COL_ACCENT),
             ("A physics model of the defensive half of a ballistic missile", COL_TEXT),
             ("engagement: detection, fire control, boost, homing, miss distance.", COL_TEXT),
             ("Every number on every screen is computed from the equations in", COL_TEXT),
             ("mode 8. None of them is stored as a result.", COL_TEXT),
             ("", COL_TEXT),
             ("WHAT IT DELIBERATELY OMITS", COL_ACCENT),
             ("Threat vehicle design, propulsion, staging, reentry-vehicle and", COL_TEXT),
             ("warhead physics, and countermeasure or penetration-aid design.", COL_TEXT),
             ("None of it is present and none of it is needed. The threat enters", COL_TEXT),
             ("as a Keplerian arc derived from ground range alone, because that", COL_TEXT),
             ("is what a defence actually has: a track, not a blueprint. There is", COL_TEXT),
             ("no manufacturing guide in this file.", COL_TEXT),
             ("", COL_TEXT),
             ("NO RAILGUN MODEL -- THE ABSENCE IS DELIBERATE", COL_ACCENT),
             ("This program assesses a railgun-launched interceptor and finds it", COL_TEXT),
             ("does not work. It does that with equations, margins and measured", COL_TEXT),
             ("correlations only. There is no geometry, no assembly, no", COL_TEXT),
             ("dimensioned drawing and no build description -- not for the", COL_TEXT),
             ("launcher, not for the projectile. The GUN view draws margin bars", COL_TEXT),
             ("and pass/fail marks and nothing that could be traced.", COL_TEXT),
             ("", COL_TEXT),
             ("A critique and a blueprint are different objects with different", COL_TEXT),
             ("reach. A critique tells you whether a concept closes; a model", COL_TEXT),
             ("tells you how to make one. Only the first answers the question", COL_TEXT),
             ("this project asks, and only the second keeps its value in hands", COL_TEXT),
             ("the author never meant to help. Engineering detail travels once", COL_TEXT),
             ("written down. It does not check who is reading it, and it does", COL_TEXT),
             ("not stop working when it arrives somewhere unwelcome --", COL_TEXT),
             ("convergent designs across supposedly independent competitors are", COL_TEXT),
             ("the visible evidence of how readily it moves.", COL_TEXT),
             ("", COL_TEXT),
             ("A negative result carries no such risk: 'this fails, and here is", COL_TEXT),
             ("the number that kills it' is not a manufacturing input. So the", COL_TEXT),
             ("salvo mathematics is present in full and the hardware is absent.", COL_TEXT),
             ("Everything needed to EVALUATE the concept is here. Nothing needed", COL_WARN),
             ("to BUILD one is, and nothing of that kind should be added later.", COL_WARN),
             ("", COL_TEXT),
             ("THE FINDING", COL_ACCENT),
             ("Hitting one object is a solved kinematics problem and the model", COL_TEXT),
             ("closes it cleanly. Every complication after that degrades the", COL_TEXT),
             ("result faster than interceptor performance recovers it, and the", COL_TEXT),
             ("one that degrades it fastest -- object count -- is a counting", COL_TEXT),
             ("problem that better hardware does not touch.", COL_TEXT),
             ("", COL_TEXT),
             ("That asymmetry is the entire content of the model, and it is why", COL_TEXT),
             ("'does missile defence work' has no single answer. Against one", COL_TEXT),
             ("unaccompanied object with good track: yes, and the geometry shows", COL_TEXT),
             ("why. Against anything else the question stops being about physics,", COL_TEXT),
             ("and this stops being the right tool for answering it.", COL_TEXT),
             ("", COL_TEXT),
             ("AUTO-AIM CALCULATION -- PROOF OF THE SENSOR-TO-INTERCEPT CHAIN", COL_ACCENT),
             ("", COL_TEXT),
             ("The auto-aim is not a lookup or a heuristic. It is a closed chain", COL_TEXT),
             ("of five equations, each feeding the next, from raw sensor data to", COL_TEXT),
             ("a terminal guidance command. Every number on every screen is", COL_TEXT),
             ("computed live from this chain -- none is stored as a result.", COL_TEXT),
             ("", COL_TEXT),
             ("1. THREAT ARC FROM GROUND RANGE (conic range equation)", COL_ACCENT),
             ("", COL_TEXT),
             ("Given only ground range R_g, the minimum-energy flight-path angle", COL_TEXT),
             ("on a spherical Earth is:", COL_TEXT),
             ("", COL_TEXT),
             ("    gamma = pi/4 - psi/4,  psi = R_g / R_E", COL_TEXT),
             ("", COL_TEXT),
             ("Proof: at psi -> 0 (flat Earth), gamma -> 45 deg, recovering the", COL_TEXT),
             ("flat-Earth projectile optimum. At psi -> pi (antipodal), gamma -> 0", COL_TEXT),
             ("and the burnout speed -> sqrt(mu/R_E) = 7.90 km/s, which is exactly", COL_TEXT),
             ("circular orbital speed -- the only trajectory that reaches the", COL_TEXT),
             ("opposite point. Both limits are verified in selftest.", COL_TEXT),
             ("", COL_TEXT),
             ("Burnout speed is then found by bisection on the EXACT conic range", COL_TEXT),
             ("equation:", COL_TEXT),
             ("", COL_TEXT),
             ("    Q = v^2 r / mu,  e = sqrt(1 + Q(Q-2) cos^2 gamma)", COL_TEXT),
             ("    p = (r v cos gamma)^2 / mu", COL_TEXT),
             ("    range = (nu_impact - nu_burnout) * R_E", COL_TEXT),
             ("", COL_TEXT),
             ("where nu at each endpoint comes from r = p/(1 + e cos nu). This is", COL_TEXT),
             ("exact for burnout at altitude, not a surface-to-surface approximation.", COL_TEXT),
             ("", COL_TEXT),
             ("2. SENSOR DETECTION (radar range equation + horizon)", COL_ACCENT),
             ("", COL_TEXT),
             ("Maximum detection range from the one-way radar equation:", COL_TEXT),
             ("", COL_TEXT),
             ("    R_max^4 = P_t G^2 lambda^2 sigma n / ((4pi)^3 k T B F (S/N))", COL_TEXT),
             ("", COL_TEXT),
             ("where n is coherent integration gain. Every term lives in SENSORS", COL_TEXT),
             ("and moves when assumptions do. The geometric horizon is:", COL_TEXT),
             ("", COL_TEXT),
             ("    R_horizon = sqrt((R_E+h_s)^2 - R_E^2) + sqrt((R_E+h_t)^2 - R_E^2)", COL_TEXT),
             ("", COL_TEXT),
             ("first_detection sweeps the trajectory and returns the earliest time", COL_TEXT),
             ("both R < R_max AND R < R_horizon AND line-of-sight clears the Earth.", COL_TEXT),
             ("Detection TIME, not radar power, sets the battlespace -- the horizon", COL_TEXT),
             ("usually binds before the range equation does.", COL_TEXT),
             ("", COL_TEXT),
             ("3. TRACK QUALITY (error growth since last update)", COL_ACCENT),
             ("", COL_TEXT),
             ("Track error at handover is:", COL_TEXT),
             ("", COL_TEXT),
             ("    sigma(t) = sigma_0 + sigma_dot * (t_intercept - t_last_update)", COL_TEXT),
             ("", COL_TEXT),
             ("Growth is charged since the last sensor UPDATE, not since first", COL_TEXT),
             ("detection. With in-flight target updates (IFTU) the gap is the final", COL_TEXT),
             ("uplink lead; without it the track is stale for the entire engagement.", COL_TEXT),
             ("This term, not guidance quality, is the dominant error source.", COL_TEXT),
             ("", COL_TEXT),
             ("4. FIRE CONTROL (universal-variable Lambert solver)", COL_ACCENT),
             ("", COL_TEXT),
             ("Given the interceptor site r1 and the threat's PREDICTED position r2", COL_TEXT),
             ("at time t_intercept, Lambert's problem returns the velocity the", COL_TEXT),
             ("interceptor must have at burnout:", COL_TEXT),
             ("", COL_TEXT),
             ("    Find z such that:  sqrt(y/C) * (x^3 S + A sqrt(y)) / sqrt(mu) = tof", COL_TEXT),
             ("    where:  y = r1 + r2 + A(z S - 1)/sqrt(C)", COL_TEXT),
             ("            A = sin(dnu) sqrt(r1 r2 / (1 - cos dnu))", COL_TEXT),
             ("            C = c2(z),  S = c3(z)  (Stumpff functions)", COL_TEXT),
             ("", COL_TEXT),
             ("Bisected on z over [-4pi^2, 4pi^2]. Both transfer directions are", COL_TEXT),
             ("tried (lambert_best) because the site sits downrange -- the short", COL_TEXT),
             ("way round is often retrograde. The velocity BOUGHT is:", COL_TEXT),
             ("", COL_TEXT),
             ("    dv = |v_lambert - v_platform|", COL_TEXT),
             ("", COL_TEXT),
             ("crediting Earth co-rotation (surface) or orbital velocity (space).", COL_TEXT),
             ("solve_intercept sweeps candidate intercept times and returns the", COL_TEXT),
             ("EARLIEST feasible PIP where dv <= budget and altitude > 120 km.", COL_TEXT),
             ("", COL_TEXT),
             ("5. TERMINAL HOMING (zero-effort-miss proportional navigation)", COL_ACCENT),
             ("", COL_TEXT),
             ("With r_rel = r_interceptor - r_target, the guidance law is:", COL_TEXT),
             ("", COL_TEXT),
             ("    t_go = -(r_rel . v_rel) / |v_rel|^2", COL_TEXT),
             ("    ZEM = r_rel + v_rel * t_go        (predicted miss vector)", COL_TEXT),
             ("    a_cmd = -N * ZEM_perp / t_go^2", COL_TEXT),
             ("", COL_TEXT),
             ("Proof of convergence: ZEM_perp is the component of ZEM", COL_TEXT),
             ("perpendicular to v_rel. The command accelerates to null it, so", COL_TEXT),
             ("d(ZEM_perp)/dt = -N/t_go * ZEM_perp + O(lag). With N > 2 and zero", COL_TEXT),
             ("lag, ZEM_perp -> 0 exponentially and the miss -> 0. Two physical", COL_TEXT),
             ("effects prevent a perfect hit:", COL_TEXT),
             ("", COL_TEXT),
             ("  (a) Guidance lag tau: achieved acceleration chases the command", COL_TEXT),
             ("      as a(t+dt) = a(t) + (a_cmd - a(t)) * dt/tau, so commands are", COL_TEXT),
             ("      never achieved instantly.", COL_TEXT),
             ("  (b) Aimpoint bias: the seeker nulls to where it THINKS the target", COL_TEXT),
             ("      is, so the kill vehicle cannot close inside its own boresight", COL_TEXT),
             ("      error. With bias Gaussian in 2 axes, miss is Rayleigh and:", COL_TEXT),
             ("", COL_TEXT),
             ("      Pk_max = 1 - exp(-R_lethal^2 / (2 sigma^2))", COL_TEXT),
             ("", COL_TEXT),
             ("This closed form is the CEILING -- no divert budget, sensor upgrade", COL_TEXT),
             ("or guidance change moves it. The Monte Carlo reproduces it, which", COL_TEXT),
             ("is the strongest single validation in this file.", COL_TEXT),
             ("", COL_TEXT),
             ("Differential gravity is computed exactly: g(interceptor) - g(target),", COL_TEXT),
             ("including J2, from the full threat trajectory. At 1,300 km altitude", COL_TEXT),
             ("the gradient is ~9e-7 /s^2, drifting two bodies 4 km apart by ~10 m", COL_TEXT),
             ("over 75 s -- twenty times the lethal radius, so it is not ignorable.", COL_TEXT),
             ("", COL_TEXT),
             ("PARAMETER PROVENANCE", COL_ACCENT),
             ("Earth, atmosphere, J2 and all orbital mechanics are standard", COL_TEXT),
             ("reference values (Vallado; Bate, Mueller and White). Interceptor", COL_TEXT),
             ("dimensions are open published figures. Performance values --", COL_TEXT),
             ("divert budget, seeker range and noise, track error growth -- are", COL_TEXT),
             ("ESTIMATES, marked est. in mode 2 and in the source. They are the", COL_TEXT),
             ("least trustworthy numbers here and they drive Pk more than", COL_TEXT),
             ("anything else, so read Pk as a shape, not as a figure.", COL_TEXT),
             ("", COL_TEXT),
             ("Nothing in this model has been validated against a real system.", COL_WARN)]
        y = rect.y + 30 - self.scroll * 8
        for txt, col in L:
            if rect.y <= y <= rect.bottom - 10:
                # Section headers get a subtle divider line
                if col == COL_ACCENT and txt:
                    div_w = self.font.size(txt)[0]
                    pg.draw.line(s, (28, 38, 56), (rect.x + 64, y + 18),
                                 (rect.x + 64 + div_w + 8, y + 18), 1)
                self._text(s, txt, (rect.x + 64, y), self.font, col)
            y += 22


# =============================================================================
# SECTION 9 -- SELFTEST AND CLI
# =============================================================================


def selftest():
    """Headless verification. Each check has an independently known correct
    answer -- these test PHYSICS, not agreement with a hoped-for result."""
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1-2. Limits of the minimum-energy solution.
    v, _ = burnout_for_range(math.pi * R_E / 1000.0 * 0.995, R_E)
    v_circ = math.sqrt(MU / R_E)
    chk("range eq: antipodal -> circular speed", abs(v - v_circ) / v_circ < 0.03,
        f"{v:.0f} vs {v_circ:.0f} m/s")
    _, g2 = burnout_for_range(50.0, R_E)
    chk("range eq: short range -> 45 deg", abs(math.degrees(g2) - 45.0) < 1.0,
        f"{math.degrees(g2):.2f} deg")

    # 3. conic_range must invert burnout_for_range.
    vv, gg = burnout_for_range(10000.0, R_E)
    back = conic_range(vv, gg, R_E)
    chk("conic range inverts cleanly", back and abs(back / 1000.0 - 10000) < 20,
        f"{(back or 0)/1000:,.1f} km")

    # 4-7. Propagated ICBM arc against published magnitudes.
    traj, meta = build_threat(10000.0)
    chk("10,000 km apogee in published band",
        1100 <= meta["apogee_km"] <= 1500, f"{meta['apogee_km']:,.0f} km")
    chk("propagated range matches request",
        abs(meta["actual_range_km"] - 10000) < 400,
        f"{meta['actual_range_km']:,.0f} km")
    chk("ICBM flight time ~30 min",
        24 <= meta["total_time_s"] / 60 <= 40,
        f"{meta['total_time_s']/60:.1f} min")
    # Energy symmetry holds at the ENTRY interface, not at the ground: the arc
    # is very nearly conservative down to 100 km, then drag takes most of it.
    chk("entry speed near burnout speed (energy symmetry)",
        abs(meta["entry_speed_ms"] - meta["v_burnout_ms"]) / meta["v_burnout_ms"] < 0.12,
        f"{meta['entry_speed_ms']:.0f} vs {meta['v_burnout_ms']:.0f} m/s")
    # And drag must actually do something on the way to the ground.
    chk("drag decelerates below the entry interface",
        meta["impact_speed_ms"] < meta["entry_speed_ms"] * 0.8,
        f"{meta['entry_speed_ms']:.0f} -> {meta['impact_speed_ms']:.0f} m/s")

    # 8-9. Atmosphere.
    chk("sea-level density", abs(atmos_density(0.0) - 1.225) < 0.01,
        f"{atmos_density(0.0):.4f} kg/m^3")
    d = [atmos_density(a) for a in np.arange(0, 900e3, 5e3)]
    chk("density monotonically decreasing",
        all(d[i] >= d[i + 1] for i in range(len(d) - 1)))

    # 10-11. Lambert must actually reproduce the target when propagated.
    r1 = np.array([R_E + 500e3, 0.0, 0.0])
    r2 = np.array([0.0, R_E + 800e3, 0.0])
    sol = lambert(r1, r2, 1800.0)
    chk("lambert returns a solution", sol is not None)
    if sol:
        # Lambert is a two-body solver, so verify it against two-body motion.
        # Propagating with J2 on drifts ~19 km over this transfer, which is a
        # real perturbation, not a solver error -- and is why fire control
        # updates in flight rather than shooting once and hoping.
        r, vv2 = r1.copy(), sol[0].copy()
        t = 0.0
        while t < 1800.0:
            step = min(1.0, 1800.0 - t)
            r, vv2 = rk4_step(r, vv2, step, use_j2=False)
            t += step
        err = float(np.linalg.norm(r - r2))
        chk("lambert propagates to its target", err < 1000.0, f"{err:,.0f} m")

    # Both transfer directions must be tried; the short way is often retrograde.
    chk("lambert_best beats single-direction solve",
        lambert_best(r1, r2, 1800.0) is not None)

    # 12-13. Radar.
    ra = radar_max_range(SENSORS["ew_radar"], 0.01)
    rb = radar_max_range(SENSORS["ew_radar"], 0.16)
    chk("radar range scales as RCS^0.25", abs(rb / ra - 2.0) < 0.02,
        f"ratio {rb/ra:.4f}")
    chk("horizon grows with altitude",
        radar_horizon_m(1200e3) > radar_horizon_m(200e3))

    # 14-15. Nominal engagement closes.
    eng = Engagement(range_km=10000.0)
    chk("nominal engagement finds a solution", eng.solution is not None)
    chk("nominal engagement computes Pk", eng.pk is not None)

    # 16-17. Pk MUST degrade with worse track. If it does not, the homing model
    # is ignoring the track error and the number on screen is decorative.
    if eng.solution:
        good = single_shot_pk(eng.solution, INTERCEPTOR, 100.0, trials=80, seed=3)
        bad = single_shot_pk(eng.solution, INTERCEPTOR, 8000.0, trials=80, seed=3)
        chk("Pk degrades with track error", bad["pk"] < good["pk"],
            f"{good['pk']*100:.0f}% -> {bad['pk']*100:.0f}%")
        # p95, not median: PN holds the median at the aimpoint floor on both
        # sides of the divert-saturation cliff, so a median-based check passes
        # even when the tail has blown out by four orders of magnitude.
        chk("miss tail grows with track error",
            bad["miss_p95"] > good["miss_p95"] * 10.0,
            f"p95 {good['miss_p95']:.2f} -> {bad['miss_p95']:,.0f} m")

    # 18. Interceptor demand must scale with object count.
    e1 = Engagement(range_km=10000.0, objects=1, trials=60)
    e10 = Engagement(range_km=10000.0, objects=10, trials=60)
    if e1.pk and e10.pk and e1.pk["pk"] > 0:
        n1, n10 = e1.interceptors_for_leakage(0.01), e10.interceptors_for_leakage(0.01)
        # Demand is linear in object count, within integer rounding: each
        # ceil() can add up to one interceptor, so 10x objects lands in
        # [10*n1 - 10, 10*n1]. Asserting exactly 10x would fail on rounding
        # alone and would be testing arithmetic, not the scaling claim.
        chk("interceptor demand scales with object count",
            n1 and n10 and (10 * n1 - 10) <= n10 <= 10 * n1,
            f"{n1} -> {n10}")

    # 19. No fabricated certainty. A model reporting Pk = 100% under noise has
    # stopped modelling the thing it claims to model.
    if eng.pk:
        chk("Pk is not fabricated at 100%", eng.pk["pk"] < 1.0,
            f"{eng.pk['pk']*100:.1f}%")

    # 20. The booster must not be magic.
    if eng.solution:
        chk("burnout dv within stated capability",
            eng.solution["dv_required"] <= INTERCEPTOR["burnout_v_ms"],
            f"{eng.solution['dv_required']:,.0f} m/s")

    # 21-23. Gun-launched proposal assessment.
    lim = railgun_material_limit()
    chk("railgun material limit reproduces Mach 12.6",
        abs(lim["v_max"] / 343.0 - 12.6) < 0.3,
        f"Mach {lim['v_max']/343.0:.2f}")
    chk("claimed muzzle velocity overstresses the projectile",
        lim["overstress"] > 3.0, f"{lim['overstress']:.1f}x limit")
    # The correlated-salvo model must reduce to the textbook formula when the
    # shots really are independent -- otherwise it is not measuring what the
    # document assumed, and the comparison would be meaningless.
    pc0, pi0 = salvo_probability(0.11, 46, 0.0)
    pc9, _ = salvo_probability(0.11, 46, 0.9)
    chk("salvo model matches 1-(1-p)^k when independent",
        abs(pc0 - pi0) < 0.01, f"{pc0*100:.1f}% vs {pi0*100:.1f}%")
    chk("correlated salvo collapses vs independent", pc9 < pi0 - 0.3,
        f"{pi0*100:.0f}% -> {pc9*100:.0f}%")

    # 24-26. Gun sandbox: the density trade must show up in both directions,
    # and nothing may pass all four checks at Mach 62.
    v62 = 62 * 343.0
    gw = gun_assessment(v62, "tungsten", 10.72)
    gd = gun_assessment(v62, "CVD diamond", 10.72)
    chk("diamond survives the barrel where tungsten does not",
        gd["pass_barrel"] and not gw["pass_barrel"],
        f"diamond {gd['stress_ratio']:.2f}x, tungsten {gw['stress_ratio']:.1f}x")
    chk("diamond keeps less velocity through the air",
        gd["v_kept"] < gw["v_kept"],
        f"{gd['v_kept']*100:.0f}% vs {gw['v_kept']*100:.0f}%")
    passes_all = []
    for mat in MATERIAL_NAMES:
        for bl in (10.72, 100.0, 300.0):
            g = gun_assessment(v62, mat, bl)
            if g["pass_barrel"] and g["pass_thermal"] and g["v_kept"] > 0.5:
                passes_all.append(f"{mat}@{bl:.0f}m")
    chk("no material passes every check at Mach 62", not passes_all,
        "; ".join(passes_all) if passes_all else
        f"0 of {len(MATERIAL_NAMES)*3} combinations")

    # Launch platforms: losses must be charged, and placement must matter.
    dv_g = platform_dv(INTERCEPTOR, "ground_silo")
    dv_a = platform_dv(INTERCEPTOR, "air_launched")
    dv_s = platform_dv(INTERCEPTOR, "space_based")
    chk("surface launch is charged gravity+drag losses",
        dv_g < INTERCEPTOR["burnout_v_ms"] - 700.0,
        f"{INTERCEPTOR['burnout_v_ms']:,.0f} -> {dv_g:,.0f} m/s")
    chk("air launch delivers more than ground launch", dv_a > dv_g + 300.0,
        f"{dv_a:,.0f} vs {dv_g:,.0f} m/s")
    chk("orbital platform pays no ascent losses",
        abs(dv_s - INTERCEPTOR["burnout_v_ms"]) < 1.0, f"{dv_s:,.0f} m/s")

    e_sp = Engagement(range_km=10000.0, platform="space_based", trials=40)
    pl = e_sp.placement
    # The orbital advantage is placement-dependent: some stations must reverse
    # their own orbital velocity to engage and are worse than a ground silo.
    # If every station worked, the model would be hiding the absentee problem.
    chk("orbital placement search finds both usable and unusable stations",
        pl and 0 < pl["n_feasible"] < pl["n_tried"],
        f"{pl['n_feasible']}/{pl['n_tried']} feasible" if pl else "no search")
    chk("space-based engagement closes from its best station",
        e_sp.solution is not None and e_sp.pk is not None,
        f"Pk {e_sp.pk['pk']*100:.0f}%" if e_sp.pk else "none")

    # Off-track geometry and defended footprint.
    e_on = Engagement(range_km=10000.0, defence_offset_km=0.0, trials=40)
    e_off = Engagement(range_km=10000.0, defence_offset_km=2000.0, trials=40)
    arc = math.asin(min(1.0, abs(e_off.site_r[2]) / R_E)) * R_E / 1000.0
    chk("off-track offset moves the site by the requested arc",
        abs(arc - 2000.0) < 30.0, f"{arc:,.0f} km of 2,000 km")
    chk("off-track site still on the Earth's surface",
        abs(float(np.linalg.norm(e_off.site_r)) - R_E) < 1.0,
        f"r = {np.linalg.norm(e_off.site_r)/1000:,.1f} km")
    # Lateral reach is nearly free from rest but not from orbit. If offsetting
    # a ground site cost orbital-style plane-change delta-v, the geometry would
    # be wrong somewhere.
    if e_on.solution and e_off.solution:
        extra = e_off.solution["dv_required"] - e_on.solution["dv_required"]
        chk("ground plane change is cheap (aim, not manoeuvre)",
            abs(extra) < 300.0, f"{extra:+.0f} m/s for 2,000 km offset")
    fp = footprint_radius_km(10000.0, iters=6)
    chk("defended footprint is finite and non-trivial",
        800.0 < fp < 6000.0, f"{fp:,.0f} km radius")
    e_far = Engagement(range_km=10000.0, defence_offset_km=fp + 1500.0, trials=4)
    chk("beyond the footprint the intercept fails",
        e_far.solution is None, "infeasible past the edge")

    # The strongest validation available: the 3D Monte Carlo must reproduce an
    # independent closed form for the aimpoint-limited ceiling. If these two
    # ever disagree, one of them is wrong and both are load-bearing.
    ceil_ok = []
    for ap in (0.1, 0.2, 0.4):
        sp = dict(INTERCEPTOR)
        sp["kv_aimpoint_sigma_m"] = ap
        sp["kv_divert_dv_ms"] = 2000.0
        r = single_shot_pk(e_on.solution, sp, 50.0, trials=400, seed=1)
        ceil_ok.append((ap, r["pk"], aimpoint_ceiling(ap)))
    worst = max(abs(s - a) for _, s, a in ceil_ok)
    chk("Monte Carlo matches the closed-form aimpoint ceiling", worst < 0.06,
        "  ".join(f"s={ap}: {s*100:.0f}/{a*100:.0f}%" for ap, s, a in ceil_ok))

    # Hermite interpolation must beat the linear sagitta it replaced.
    trj, _ = build_threat(10000.0, dt=1.0)
    i = 800
    t_mid = float(trj.t[i]) + 0.5
    r_h, _ = trj.at(t_mid)
    rr, vv = trj.r[i].copy(), trj.v[i].copy()
    tt = float(trj.t[i])
    while tt < t_mid:
        st = min(0.005, t_mid - tt)
        rr, vv = rk4_step(rr, vv, st, beta=THREAT_BETA)
        tt += st
    lin = 0.5 * (trj.r[i] + trj.r[i+1])
    chk("Hermite interpolation beats linear on a curved arc",
        float(np.linalg.norm(rr - r_h)) < float(np.linalg.norm(rr - lin)) / 5.0,
        f"{np.linalg.norm(rr-r_h):.3f} m vs linear {np.linalg.norm(rr-lin):.3f} m")

    # In-flight updates must be neutral at good sensor quality and decisive at bad.
    _gw = TRACK["growth_m_per_s"]
    try:
        TRACK["growth_m_per_s"] = 20.0
        a_bad = Engagement(range_km=10000.0, iftu=False, trials=200)
        b_bad = Engagement(range_km=10000.0, iftu=True, trials=200)
        chk("in-flight updates rescue a degraded sensor",
            a_bad.pk and b_bad.pk and b_bad.pk["pk"] > a_bad.pk["pk"] + 0.35,
            f"{a_bad.pk['pk']*100:.0f}% -> {b_bad.pk['pk']*100:.0f}%")
        chk("updates hold track error near the uplink lead",
            b_bad.track_sigma < a_bad.track_sigma / 10.0,
            f"{a_bad.track_sigma:,.0f} -> {b_bad.track_sigma:,.0f} m")
    finally:
        TRACK["growth_m_per_s"] = _gw

    # Monte Carlo estimates must carry their sampling error.
    small = single_shot_pk(e_on.solution, INTERCEPTOR, 3000.0, trials=30, seed=2)
    big = single_shot_pk(e_on.solution, INTERCEPTOR, 3000.0, trials=400, seed=2)
    chk("Pk confidence interval narrows with trials",
        small["pk_halfwidth"] > big["pk_halfwidth"] * 1.8,
        f"+/-{small['pk_halfwidth']*100:.0f} pts at 30 -> "
        f"+/-{big['pk_halfwidth']*100:.0f} pts at 400")
    chk("Wilson interval stays inside [0,1] at the extremes",
        wilson_interval(0, 50)[0] >= 0.0 and wilson_interval(50, 50)[1] <= 1.0,
        f"{wilson_interval(0,50)[0]:.3f} .. {wilson_interval(50,50)[1]:.3f}")

    # 3D vehicle geometry must match the spec it claims to be drawn from.
    mesh = build_interceptor_mesh()
    allv = np.vstack([q["verts"] for q in mesh])
    body = allv[allv[:, 2] >= 0.0]
    length_mm = float(body[:, 2].max())
    chk("3D stack length matches spec exactly",
        abs(length_mm - INTERCEPTOR["length_mm"]) < 1.0,
        f"{length_mm:,.0f} vs {INTERCEPTOR['length_mm']:,.0f} mm")
    # Radius check excludes fin extensions (fins intentionally project past
    # the body diameter); check only the cylindrical/frustum body vertices.
    R_spec = INTERCEPTOR["diameter_mm"] / 2.0
    body_mask = np.hypot(allv[:, 0], allv[:, 1]) <= R_spec + 1.0
    body_verts = allv[body_mask]
    rad = float(np.hypot(body_verts[:, 0], body_verts[:, 1]).max())
    chk("3D stack radius matches spec",
        abs(rad - R_spec) < 1.0,
        f"{rad:,.0f} vs {R_spec:,.0f} mm")

    # The kill vehicle is factored into its own mesh (base at origin, nose at
    # +Z) so the terminal collision view can place and orient it without
    # dragging the spent booster along. It must still match the KV portion of
    # the full stack exactly, or the two views have silently drifted apart.
    kv = build_kv_mesh()
    kv_all = np.vstack([q["verts"] for q in kv])
    chk("KV mesh base sits at local origin",
        abs(float(kv_all[:, 2].min())) < 1e-6, f"{kv_all[:,2].min():.6f} mm")
    chk("KV mesh length matches spec",
        abs(float(kv_all[:, 2].max()) - (INTERCEPTOR["kv_len_mm"] + 240.0)) < 1.0,
        f"{kv_all[:,2].max():.0f} mm")
    stack_kv = [p for p in build_interceptor_mesh() if p["group"] == 4]
    kz = float(stack_kv[0]["verts"][:, 2].min())
    chk("standalone KV mesh matches the KV embedded in the full stack",
        all(np.allclose(a["verts"] - np.array([0, 0, kz]), b["verts"])
            for a, b in zip(stack_kv, kv)),
        "vertex-identical after re-basing")

    # --- terminal collision: the picture must be the physics, not a roll ---
    # This is the coverage that would have caught a fixed-seed coin flip
    # standing in for the simulated intercept: it checks that the ENDPOINT of
    # the rendered path is the same point homing_run actually computed, not
    # merely that hit/miss was set to something.
    zeng = Engagement(range_km=10000.0, objects=1, trials=20)
    zapp = App.__new__(App)      # construct without opening a pygame window
    zapp.args = argparse.Namespace(threat_range=10000.0, loft=1.0, objects=1)
    zapp.eng = zeng
    path = zapp._interceptor_path()
    chk("terminal path splice produced a result",
        path is not None and zapp.term_hit is not None)
    if path is not None:
        ip, it = path
        target_at_cpa, _ = zeng.traj.at(zapp.term_cpa_t)
        end_gap = float(np.linalg.norm(ip[-1] - target_at_cpa))
        chk("rendered path endpoint matches the recorded miss distance",
            abs(end_gap - zapp.term_miss_m) < 1e-2,
            f"path {end_gap:.4f} m vs recorded {zapp.term_miss_m:.4f} m")
        chk("hit flag agrees with lethal radius threshold",
            zapp.term_hit == (zapp.term_miss_m <= zeng.spec["kv_lethal_radius_m"]))

        zapp2 = App.__new__(App)
        zapp2.args = zapp.args
        zapp2.eng = zeng
        zapp2._interceptor_path()
        chk("terminal seed is deterministic across rebuilds",
            abs(zapp2.term_miss_m - zapp.term_miss_m) < 1e-9,
            f"{zapp.term_miss_m:.6f} vs {zapp2.term_miss_m:.6f} m")

    # And it must actually respond to degraded track quality -- proving the
    # outcome is wired to the same physics --feasibility quotes, not frozen.
    _gw = TRACK["growth_m_per_s"]
    try:
        TRACK["growth_m_per_s"] = 30.0
        bad_eng = Engagement(range_km=10000.0, objects=1, iftu=False, trials=20)
    finally:
        TRACK["growth_m_per_s"] = _gw
    bapp = App.__new__(App)
    bapp.args = zapp.args
    bapp.eng = bad_eng
    bapp._interceptor_path()
    chk("degraded track quality can flip the terminal outcome to a miss",
        bapp.term_hit is False and bapp.term_miss_m > 100.0,
        f"miss {bapp.term_miss_m:,.0f} m (track sigma {bad_eng.track_sigma:,.0f} m)")
    chk("divert saturates at the stated budget in the bad-track case",
        abs(bapp.term_dv_used - bad_eng.spec["kv_divert_dv_ms"]) < 1.0,
        f"{bapp.term_dv_used:.1f} of {bad_eng.spec['kv_divert_dv_ms']:.0f} m/s")

    # --- salvo / multiple kill vehicles ---
    seng = Engagement(range_km=10000.0, trials=20)
    # k=1 must reduce exactly to the single-vehicle model.
    s1 = salvo_kill_probability(seng.solution, seng.spec, 3000.0, k=1,
                                trials=200, seed=4)
    chk("salvo model reduces to single-shot at k=1",
        abs(s1["pk"] - s1["p_per_vehicle"]) < 1e-9,
        f"{s1['pk']*100:.1f}% vs {s1['p_per_vehicle']*100:.1f}%")
    # P(kill) must be monotonically non-decreasing in k. It is mathematically
    # impossible for a larger salvo to do worse -- the salvo takes the best of
    # k -- so a decrease means the sampling is not paired across k, which is
    # exactly the artefact that produced 4 vehicles scoring below 1.
    seq = [salvo_kill_probability(seng.solution, seng.spec, 20000.0, k=kk,
                                  trials=160, seed=8)["pk"]
           for kk in (1, 2, 3, 4)]
    chk("salvo P(kill) is monotone in vehicle count",
        all(seq[i] <= seq[i+1] + 1e-12 for i in range(len(seq) - 1)),
        " -> ".join(f"{v*100:.1f}%" for v in seq))

    # The independence claim must be measured, not assumed: vehicles should be
    # uncorrelated when aimpoint-limited and strongly correlated when the
    # shared track error dominates.
    c_good = salvo_correlation(seng.solution, seng.spec, 270.0, trials=400,
                               seed=11)
    c_bad = salvo_correlation(seng.solution, seng.spec, 20000.0, trials=400,
                              seed=11)
    chk("kill vehicles fail independently when aimpoint-limited",
        abs(c_good["phi"]) < 0.25, f"phi = {c_good['phi']:+.2f}")
    chk("kill vehicles fail together when track-limited",
        c_bad["phi"] > 0.6, f"phi = {c_bad['phi']:+.2f}")
    chk("joint miss rate exceeds independence when correlated",
        c_bad["joint_miss"] > c_bad["joint_if_independent"] + 0.05,
        f"{c_bad['joint_miss']*100:.0f}% vs {c_bad['joint_if_independent']*100:.0f}% independent")

    # Salvo must actually help when aimpoint-limited...
    g1 = salvo_kill_probability(seng.solution, seng.spec, 270.0, k=1,
                                trials=200, seed=6)
    g2 = salvo_kill_probability(seng.solution, seng.spec, 270.0, k=2,
                                trials=200, seed=6)
    chk("extra kill vehicles raise Pk when aimpoint-limited",
        g2["pk"] > g1["pk"] + 0.01,
        f"{g1['pk']*100:.1f}% -> {g2['pk']*100:.1f}%")

    # ...and must be worthless once divert saturates, because every vehicle
    # burns its whole budget flying toward the same wrong place. This is the
    # sharp end of the correlation argument and the one most likely to be
    # softened by accident into "helps a bit".
    sat_sigma = 20000.0
    track_rng = np.random.default_rng(8)
    veh = [np.random.default_rng([8, i]) for i in range(4)]
    off = track_rng.normal(0.0, sat_sigma, 2)
    voff = track_rng.normal(0.0, TRACK["velocity_sigma_ms"], 2)
    sat_miss = [homing_run(seng.solution, seng.spec, sat_sigma, veh[i],
                           shared_track=(off, voff))[0] for i in range(4)]
    chk("saturated salvo lands as one cluster (extra vehicles add nothing)",
        (max(sat_miss) - min(sat_miss)) < 25.0 and min(sat_miss) > 1000.0,
        f"spread {max(sat_miss)-min(sat_miss):.1f} m at {min(sat_miss):,.0f} m miss")

    # The railgun omission is a documented property of this program, so it is
    # enforced rather than trusted. A drawing is built from filled shapes; the
    # gun view is allowed bars, rules and tick marks and nothing else. If a
    # later change starts rendering gun or projectile geometry, this fails.
    try:
        _src = open(__file__, encoding="utf-8").read()
        _gun = _src[_src.index("def _mode_gun"):_src.index("def _mode_layered")]
        _shapes = _gun.count("pg.draw.polygon") + _gun.count("pg.draw.circle") \
            + _gun.count("gfxdraw.aapolygon") + _gun.count("gfxdraw.filled")
        chk("gun view draws no traceable geometry", _shapes == 0,
            f"{_shapes} shape-fill calls (bars/rules only are permitted)")
        for _doc, _label in ((_src[:6000], "module docstring"),):
            chk(f"{_label} states why the railgun model is absent",
                "NO RAILGUN MODEL" in _doc,
                "omission documented at the top of the file")
    except Exception as ex:
        chk("railgun omission is enforced", False, f"{type(ex).__name__}: {ex}")

    # --- salvo escalation: 'on a miss, add one more' ---
    esc_good = adaptive_salvo_size(seng.solution, seng.spec, 270.0,
                                   target_pk=0.99, max_k=8, trials=160,
                                   seed=21)
    chk("escalation terminates quickly when aimpoint-limited",
        esc_good["k"] is not None and esc_good["k"] <= 3,
        f"met target at k={esc_good['k']}")
    esc_bad = adaptive_salvo_size(seng.solution, seng.spec, 20000.0,
                                  target_pk=0.99, max_k=8, trials=160,
                                  seed=21)
    chk("escalation stalls rather than running away when track-limited",
        esc_bad["stalled"] and esc_bad["k"] is None,
        esc_bad["reason"])

    # The outcome split explains the stall and must partition cleanly.
    sp_good = salvo_outcome_split(seng.solution, seng.spec, 270.0, k=6,
                                  trials=150, seed=21)
    sp_bad = salvo_outcome_split(seng.solution, seng.spec, 40000.0, k=6,
                                 trials=150, seed=21)
    chk("outcome split partitions to 1",
        abs(sp_bad["all_hit"] + sp_bad["none_hit"] + sp_bad["mixed"] - 1.0) < 1e-9,
        f"{sp_bad['all_hit']:.3f}+{sp_bad['none_hit']:.3f}+{sp_bad['mixed']:.3f}")
    chk("whole-salvo failures grow as the shared error grows",
        sp_bad["none_hit"] > sp_good["none_hit"] + 0.5,
        f"{sp_good['none_hit']*100:.0f}% -> {sp_bad['none_hit']*100:.0f}% none-hit")

    # --- terminal perturbations: differential gravity ---
    # Both bodies are in free fall, so what bends the RELATIVE path is the
    # difference in the gravity each feels. Omitting it entirely (as this
    # model originally did) is only defensible if it is shown to be absorbed,
    # not assumed to be small -- at 4 km separation it is worth ~10 m of free
    # drift against a 0.5 m lethal radius.
    g_sol = seng.solution
    g_nog = dict(g_sol)
    g_nog.pop("traj", None)
    chk("solution carries the trajectory for differential gravity",
        g_sol.get("traj") is not None)
    # Unguided, the perturbation must visibly move the endpoint.
    spec_free = dict(seng.spec)
    spec_free.update(kv_nav_gain=0.0, kv_divert_dv_ms=0.0,
                     kv_aimpoint_sigma_m=0.0, kv_seeker_noise_urad=0.0)
    m_off, _ = homing_run(g_nog, spec_free, 4000.0,
                          np.random.default_rng(1), noise=False)
    m_on, _ = homing_run(g_sol, spec_free, 4000.0,
                         np.random.default_rng(1), noise=False)
    chk("differential gravity actually perturbs the unguided path",
        (m_on - m_off) > 20.0, f"{m_off:,.1f} -> {m_on:,.1f} m (+{m_on-m_off:,.1f})")
    # Guided, PN must absorb it -- paid for in divert, not in miss distance.
    a_g = single_shot_pk(g_nog, seng.spec, 4000.0, trials=150, seed=7)
    b_g = single_shot_pk(g_sol, seng.spec, 4000.0, trials=150, seed=7)
    chk("proportional navigation absorbs the gravity gradient",
        abs(b_g["miss_median"] - a_g["miss_median"]) < 0.05,
        f"median miss {a_g['miss_median']:.3f} -> {b_g['miss_median']:.3f} m")

    # Per-vehicle outcomes: a salvo is not one pass/fail event.
    _kvc = INTERCEPTOR["kv_count"]
    try:
        INTERCEPTOR["kv_count"] = 3
        vapp = App.__new__(App)
        vapp.args = argparse.Namespace(threat_range=10000.0, loft=1.0,
                                       objects=1)
        vapp.eng = Engagement(range_km=10000.0, objects=1, trials=20)
        vapp._interceptor_path()
        lethal = vapp.eng.spec["kv_lethal_radius_m"]
        ok = all(hit == (miss <= lethal)
                 for _p, _t, miss, hit in vapp.term_kv_paths)
        chk("each salvo vehicle carries its own hit/miss outcome",
            ok and len(vapp.term_kv_paths) == 2,
            f"{len(vapp.term_kv_paths)} secondary vehicles, flags consistent")
    finally:
        INTERCEPTOR["kv_count"] = _kvc

    # --- globe view must reflect the SAME terminal outcome as everything
    # else, not just the close-up. This is the exact regression a user
    # caught by eye: the salvo and the real hit/miss result existed in
    # every part of the program except the view most people look at first.
    try:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame as _pg
        _pg.init()
        import argparse as _ap
        rect = _pg.Rect(0, 54, SCREEN_W, SCREEN_H - 80)

        def _paint(app):
            app.mode = MODES.index("ENGAGEMENT")
            app.zoom_intercept = False
            app.screen.fill((0, 0, 0))
            app._mode_engagement(app.screen, rect)
            return np.array(_pg.surfarray.array3d(app.screen))

        def _count(arr, col, tol=20, xr=(0, 900), yr=(54, 170)):
            region = arr[xr[0]:xr[1], yr[0]:yr[1]]
            return int(np.all(np.abs(region.astype(int) - np.array(col))
                              < tol, axis=2).sum())

        gapp = App(_ap.Namespace(threat_range=10000.0, loft=1.0, objects=1))
        gapp.sim_t = gapp.term_cpa_t + 0.1
        a_hit = _paint(gapp)
        chk("globe view marks a HIT in the outcome colour",
            _count(a_hit, (255, 206, 84)) > 5,
            f"{_count(a_hit,(255,206,84))} gold px in header")
        red_after = _count(a_hit, (232, 88, 72), xr=(0, SCREEN_W),
                           yr=(54, SCREEN_H - 26))
        chk("threat marker disappears from the globe on a hit",
            red_after == 0, f"{red_after} threat-red px remain")

        _gw = TRACK["growth_m_per_s"]
        try:
            TRACK["growth_m_per_s"] = 200.0

            class _MissApp(App):
                def make_engagement(self):
                    return Engagement(range_km=10000.0, objects=1,
                                      iftu=False, trials=25)
            mapp = _MissApp(_ap.Namespace(threat_range=10000.0, loft=1.0,
                                          objects=1))
            chk("forced-degraded track produces a genuine miss for this check",
                mapp.term_hit is False, f"term_hit={mapp.term_hit}")
            mapp.sim_t = mapp.term_cpa_t + 0.1
            a_miss = _paint(mapp)
            chk("globe view marks a MISS in the outcome colour",
                _count(a_miss, (255, 148, 84)) > 5,
                f"{_count(a_miss,(255,148,84))} orange px in header")
            chk("threat marker survives on the globe when the salvo misses",
                _count(a_miss, (232, 88, 72), xr=(0, SCREEN_W),
                      yr=(54, SCREEN_H - 26)) > 0)
        finally:
            TRACK["growth_m_per_s"] = _gw

        _kvc = INTERCEPTOR["kv_count"]
        try:
            # NOT a pixel test. Seconds before closest approach the salvo has
            # already converged to sub-metre spread (measured: 0.4-1.0 m at
            # 5 s to CPA), which from a globe-distance camera projects to
            # ~0.00002 px of screen separation -- truly invisible, and making
            # it visible would mean inflating vehicle separation the same way
            # this file refuses to inflate vehicle SIZE. So this checks that
            # the draw call for each salvo vehicle actually fires -- a
            # structural check of the code path, not of something that
            # should be visible at this zoom level.
            calls = []
            _orig_polyline = App._polyline
            App._polyline = lambda self, s, pts, view, col, width=2: \
                calls.append(1)

            INTERCEPTOR["kv_count"] = 5
            a5 = App(_ap.Namespace(threat_range=10000.0, loft=1.0,
                                   objects=1))
            a5.sim_t = max(0.0, a5.term_cpa_t - 5.0)
            calls.clear()
            _paint(a5)
            n5 = len(calls)

            App._polyline = _orig_polyline
            INTERCEPTOR["kv_count"] = 1
            a1 = App(_ap.Namespace(threat_range=10000.0, loft=1.0,
                                   objects=1))
            a1.sim_t = max(0.0, a1.term_cpa_t - 5.0)
            calls.clear()
            App._polyline = lambda self, s, pts, view, col, width=2: \
                calls.append(1)
            _paint(a1)
            n1 = len(calls)
            App._polyline = _orig_polyline

            chk("each salvo vehicle issues its own draw call on the globe",
                n5 - n1 == 4, f"_polyline calls: kv=1 -> {n1}, kv=5 -> {n5} "
                              f"(expect +4)")
        finally:
            App._polyline = _orig_polyline
            INTERCEPTOR["kv_count"] = _kvc
        _pg.quit()
    except Exception as ex:
        chk("globe view reflects terminal outcome and salvo", False,
            f"{type(ex).__name__}: {ex}")

    # An all-hit Monte Carlo result must not be reported as certainty.
    chk("all-hit Monte Carlo reports a bound, not 100%",
        pct_mc(dict(pk=1.0, pk_lo=0.985, trials=260)).startswith(">"),
        pct_mc(dict(pk=1.0, pk_lo=0.985, trials=260)))

    # 29. Percentage formatter must never fabricate certainty.
    chk("formatter never rounds to 100%",
        _pct(0.99999969) != "100" and "100." not in _pct(0.99999969),
        f"0.99999969 -> {_pct(0.99999969)}%")

    # 29-30. Render sanity, headless. Every view mode must draw without
    # raising -- _draw() swallows exceptions into an on-screen banner, so a
    # broken panel would otherwise ship silently and only be found by eye.
    try:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        import pygame.gfxdraw
        pygame.init()
        q = Camera().project(np.array([R_E, 0.0, 0.0]),
                             pygame.Rect(0, 0, SCREEN_W, SCREEN_H))
        chk("camera projects a surface point", q is not None)

        import argparse as _ap
        app = App(_ap.Namespace(threat_range=10000.0, loft=1.0, objects=4))
        app.sim_t = 900.0
        broken = []
        for i, name in enumerate(MODES):
            app.mode = i
            app.screen.fill((0, 0, 0))
            try:
                getattr(app, "_mode_" + name.lower())(
                    app.screen, pygame.Rect(0, 54, SCREEN_W, SCREEN_H - 80))
            except Exception as ex:
                broken.append(f"{name}: {type(ex).__name__}")
        chk("all view modes render", not broken,
            "; ".join(broken) if broken else f"{len(MODES)}/{len(MODES)} modes")

        # The terminal collision view specifically -- approach, closest
        # approach, and past it, for BOTH a hit and a miss. This is the exact
        # coverage that was missing when the render sweep above passed while
        # the collision determination underneath it was a fixed coin flip.
        engage_i = MODES.index("ENGAGEMENT")
        app.mode = engage_i
        app.zoom_intercept = True
        rect = pygame.Rect(0, 54, SCREEN_W, SCREEN_H - 80)
        zoom_broken = []
        for label, t_off in (("approach", -3.0), ("cpa", 0.0),
                             ("past-cpa", 3.0), ("well-past", 40.0)):
            app.sim_t = max(0.0, (app.term_cpa_t or 0.0) + t_off)
            app.screen.fill((0, 0, 0))
            try:
                app._mode_engagement(app.screen, rect)
            except Exception as ex:
                zoom_broken.append(f"{label}: {type(ex).__name__}")
        chk("terminal collision view renders through approach and CPA",
            not zoom_broken,
            "; ".join(zoom_broken) if zoom_broken else "4/4 frames")

        # Drive the ACTUAL main-loop body in every mode. Rendering each mode
        # in isolation is not the same test: the launch crash that motivated
        # this check sat in run(), which the suite never entered.
        loop_broken = []
        for i, name in enumerate(MODES):
            app.mode = i
            for _ in range(3):
                try:
                    app.tick(1.0 / 60.0)
                except Exception as ex:
                    loop_broken.append(f"{name}: {type(ex).__name__}: {ex}")
                    break
        chk("main loop runs in every view mode", not loop_broken,
            "; ".join(loop_broken[:3]) if loop_broken
            else f"{len(MODES)} modes x 3 frames")

        # The self-running tour must visit every mode and stop when the user
        # intervenes -- a demo that fights for control is worse than none.
        app.mode = 0
        app.demo = True
        app.demo_t = 0.0
        app._demo_enter_mode()
        visited = set()
        tour_err = []
        for _ in range(int(app.demo_hold * (len(MODES) + 1) * 61)):
            try:
                app.tick(1.0 / 60.0)
            except Exception as ex:
                tour_err.append(f"{MODES[app.mode]}: {type(ex).__name__}: {ex}")
                break
            visited.add(MODES[app.mode])
            if len(visited) == len(MODES):
                break
        chk("demo tour reaches every mode without error",
            not tour_err and len(visited) == len(MODES),
            "; ".join(tour_err) if tour_err
            else f"{len(visited)}/{len(MODES)} modes")
        app.mode = 0
        app.demo = True
        app._event(type("E", (), {"type": pygame.KEYDOWN,
                                  "key": pygame.K_TAB})())
        chk("demo yields control when the user acts", app.demo is False,
            "TAB cancels the tour")

        # Layout must survive a small window -- the tab strip used to run at a
        # fixed pitch straight through the HUD.
        layout_err = []
        for w, h in ((1600, 950), (1100, 700), (820, 600), (640, 480)):
            try:
                app.screen = pygame.display.set_mode((w, h))
                app.tick(1.0 / 60.0)
            except Exception as ex:
                layout_err.append(f"{w}x{h}: {type(ex).__name__}")
        chk("UI renders across window sizes", not layout_err,
            "; ".join(layout_err) if layout_err else "4 sizes down to 640x480")
        pygame.quit()
    except Exception as ex:
        chk("render sanity", False, f"{type(ex).__name__}: {ex}")

    # --- Shot ladder checks ---
    try:
        eng = Engagement(range_km=10000.0, trials=60)
        chk("shot ladder builds for a standard engagement",
            eng.shot_ladder is not None and len(eng.shot_ladder) > 0,
            f"{len(eng.shot_ladder) if eng.shot_ladder else 0} shots")
        if eng.shot_ladder:
            n_early = sum(1 for s in eng.shot_ladder if s["shot_type"] == "early")
            n_probe = sum(1 for s in eng.shot_ladder if s["shot_type"] == "probe")
            n_salvo = sum(1 for s in eng.shot_ladder if s["shot_type"] == "salvo")
            n_trail = sum(1 for s in eng.shot_ladder if s["shot_type"] == "trailing")
            chk("shot ladder has early/probe or trailing shots around salvo",
                n_early + n_probe + n_trail > 0,
                f"{n_early} early, {n_probe} probe, {n_salvo} salvo, {n_trail} trailing")
            chk("shot ladder has exactly one salvo",
                n_salvo == 1, f"{n_salvo} salvo shots")
            chk("trailing shots are single-vehicle only",
                all(s["kv_count"] == 1 for s in eng.shot_ladder
                    if s["shot_type"] == "trailing"),
                f"{n_trail} trailing shots")
            chk("shot ladder is time-ordered",
                all(eng.shot_ladder[i]["t_intercept"] <=
                    eng.shot_ladder[i+1]["t_intercept"]
                    for i in range(len(eng.shot_ladder) - 1)),
                "sorted")
            leak = layered_leakage(eng.shot_ladder, eng.objects)
            single_leak = 1.0 - eng.pk["pk"]
            chk("layered leakage is lower than single-shot leakage",
                leak < single_leak,
                f"layered {leak*100:.3f}% vs single {single_leak*100:.3f}%")
            chk("each shot has a real Pk estimate",
                all(0.0 <= s["pk"]["pk"] <= 1.0 for s in eng.shot_ladder),
                "all Pk in [0,1]")
    except Exception as ex:
        chk("shot ladder builds for a standard engagement", False,
            f"{type(ex).__name__}: {ex}")

    print(BAR)
    print(" SELFTEST")
    print(BAR)
    for name, good, detail in checks:
        print(f"  [{'PASS' if good else 'FAIL'}]  {name:<44} {detail}")
    print(SUB)
    n_ok = sum(1 for _, g, _ in checks if g)
    print(f"  {n_ok}/{len(checks)} checks passed")
    print(BAR)
    return n_ok == len(checks)


def main():
    ap = argparse.ArgumentParser(
        description="ICBMI -- ballistic missile defence intercept digital twin")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--engage", action="store_true")
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--battlespace", action="store_true")
    ap.add_argument("--discrimination", action="store_true")
    ap.add_argument("--architectures", action="store_true")
    ap.add_argument("--escalate", action="store_true",
                    help="test the rule: on a miss, add one more round")
    ap.add_argument("--mkv", action="store_true",
                    help="multiple kill vehicles: when extra rounds help")
    ap.add_argument("--levers", action="store_true",
                    help="which knobs actually move Pk, others held ideal")
    ap.add_argument("--iftu", action="store_true",
                    help="value of in-flight target updates vs sensor quality")
    ap.add_argument("--footprint", action="store_true",
                    help="defended area per site vs threat range")
    ap.add_argument("--platforms", action="store_true",
                    help="compare ground / sea / air / space launch platforms")
    ap.add_argument("--layered", action="store_true",
                    help="independent engagement chains vs one-battery salvo")
    ap.add_argument("--layered-shots", action="store_true",
                    help="early single/dual shots at 500 km intervals, main salvo, trailing shots")
    ap.add_argument("--railgun", action="store_true",
                    help="assess the gun-launched proposal in INFORNMATIONAL.md")
    ap.add_argument("--montecarlo", type=int, default=0)
    ap.add_argument("--threat-range", type=float, default=10000.0,
                    dest="threat_range", help="threat ground range, km")
    ap.add_argument("--loft", type=float, default=1.0,
                    help="1.0 = minimum energy, >1 lofted")
    ap.add_argument("--objects", type=int, default=1,
                    help="objects on track the defence cannot separate")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.engage:
        report_engagement(Engagement(range_km=args.threat_range, loft=args.loft,
                                     objects=args.objects))
        return
    if args.feasibility:
        report_feasibility(args.threat_range)
        return
    if args.battlespace:
        report_battlespace()
        return
    if args.discrimination:
        report_discrimination(args.threat_range)
        return
    if args.architectures:
        report_architectures(args.threat_range)
        return
    if args.railgun:
        report_railgun()
        return
    if args.escalate:
        report_escalate(args.threat_range)
        return
    if args.mkv:
        report_mkv(args.threat_range)
        return
    if args.levers:
        report_levers(args.threat_range)
        return
    if args.iftu:
        report_iftu(args.threat_range)
        return
    if args.footprint:
        report_footprint()
        return
    if args.platforms:
        report_platforms(args.threat_range)
        return
    if args.layered:
        report_layered(args.threat_range, objects=args.objects)
        return
    if args.layered_shots:
        report_layered_shots(args.threat_range, objects=args.objects)
        return
    if args.montecarlo:
        eng = Engagement(range_km=args.threat_range, loft=args.loft)
        if not eng.solution:
            print("no feasible intercept")
            return
        r = single_shot_pk(eng.solution, INTERCEPTOR, eng.track_sigma,
                           trials=args.montecarlo, seed=1)
        print(f" {args.montecarlo} engagements at "
              f"{eng.track_sigma:,.0f} m track error")
        print(f"   Pk ............. {r['pk']*100:.1f}%")
        print(f"   miss median .... {r['miss_median']:.2f} m")
        print(f"   miss p95 ....... {r['miss_p95']:.2f} m")
        print(f"   divert p95 ..... {r['dv_p95']:.0f} m/s of "
              f"{INTERCEPTOR['kv_divert_dv_ms']:.0f}")
        return

    App(args).run()


if __name__ == "__main__":
    main()
