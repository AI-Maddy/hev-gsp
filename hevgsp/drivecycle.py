"""Synthetic drive cycles, road geometry and road-roughness profiles.

Speed profiles are built from randomized micro-trips (accelerate, cruise with
Ornstein-Uhlenbeck speed wander, decelerate, idle). The 'mixed' cycle follows
the four-phase structure of WLTC Class 3 (low / medium / high / extra-high
phase speed ranges) but is NOT the regulatory speed table.
Road roughness follows the ISO 8608 displacement PSD G_d(n) = G_d(n0)(n/n0)^-2.
"""
from dataclasses import dataclass
import numpy as np

KMH = 1 / 3.6

CYCLE_SPECS = {
    #          vmax range km/h, idle s,   cruise s,   accel,      decel
    "urban":    dict(v=(22, 55),  idle=(8, 35),  cruise=(8, 45),   acc=(0.8, 1.6), dec=(0.8, 1.6), stop_p=0.75),
    "suburban": dict(v=(45, 82),  idle=(5, 20),  cruise=(25, 110), acc=(0.6, 1.3), dec=(0.6, 1.3), stop_p=0.45),
    "highway":  dict(v=(95, 128), idle=(3, 8),   cruise=(60, 220), acc=(0.4, 1.0), dec=(0.4, 1.0), stop_p=0.05),
    "hilly":    dict(v=(40, 78),  idle=(5, 20),  cruise=(25, 110), acc=(0.5, 1.2), dec=(0.6, 1.3), stop_p=0.40),
}
WLTC_LIKE_PHASES = [  # (duration s, vmax range km/h, spec used for dynamics)
    (300, (20, 56), "urban"), (300, (35, 76), "urban"),
    (300, (55, 97), "suburban"), (300, (90, 131), "highway"),
]
ROAD_CLASS_Gd = {"A": 16e-6, "B": 64e-6, "C": 256e-6}
CYCLE_ROAD = {  # probability of ISO class per 200-800 m patch
    "urban": {"A": .2, "B": .5, "C": .3}, "suburban": {"A": .4, "B": .5, "C": .1},
    "highway": {"A": .7, "B": .3, "C": .0}, "hilly": {"A": .3, "B": .5, "C": .2},
    "mixed": {"A": .4, "B": .45, "C": .15},
}


@dataclass
class Route:
    name: str
    dt: float
    t: np.ndarray
    v_ref: np.ndarray        # m/s
    a_lat_des: np.ndarray    # m/s^2 (signed), desired lateral acceleration
    ds: float
    s: np.ndarray            # spatial grid (m)
    grade: np.ndarray        # rad, along s
    zL: np.ndarray           # left track road height (m), along s
    zR: np.ndarray
    P_aux12: np.ndarray      # W, along t
    solar: np.ndarray        # W, along t
    T_amb: float


def _ou(n, dt, tau, sigma, rng):
    x = np.zeros(n)
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1 - a * a)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * e[i]
    return x


def _segment_profile(spec, duration, dt, rng, vrange=None, start_v=0.0):
    """Micro-trip speed profile of a given duration (s)."""
    vr = vrange or spec["v"]
    v = [start_v]
    t_total = 0.0
    cur = start_v
    while t_total < duration:
        target = rng.uniform(*vr) * KMH
        acc = rng.uniform(*spec["acc"])
        # accelerate / decelerate to target
        while abs(cur - target) > 1e-3:
            step = np.clip(target - cur, -acc * dt, acc * dt)
            cur += step
            v.append(cur)
        cru = rng.uniform(*spec["cruise"])
        n = int(cru / dt)
        wander = _ou(n, dt, 12.0, 0.022 * target + 0.25, rng)
        v.extend(np.maximum(cur + wander, 0.5))
        cur = v[-1]
        if rng.random() < spec["stop_p"]:
            dec = rng.uniform(*spec["dec"])
            while cur > 0:
                cur = max(cur - dec * dt, 0.0)
                v.append(cur)
            v.extend([0.0] * int(rng.uniform(*spec["idle"]) / dt))
        t_total = len(v) * dt
    return np.array(v[: int(duration / dt)])


def _smooth(x, dt, win_s):
    n = max(int(win_s / dt), 1)
    k = np.ones(n) / n
    pad = np.pad(x, (n, n), mode="edge")
    return np.convolve(pad, k, mode="same")[n:-n]


def _iso8608(n_pts, ds, Gd, rng):
    """Road height with ISO 8608 PSD, band 0.011-2.83 cycles/m."""
    N = 1 << int(np.ceil(np.log2(n_pts)))
    nf = np.fft.rfftfreq(N, ds)
    G = np.zeros_like(nf)
    band = (nf >= 0.011) & (nf <= 2.83)
    G[band] = Gd * (nf[band] / 0.1) ** -2
    dn = nf[1] - nf[0]
    amp = np.sqrt(G * dn)                       # one-sided PSD -> amplitude
    phase = rng.uniform(0, 2 * np.pi, len(nf))
    spec = amp * np.exp(1j * phase) * N / 2 * np.sqrt(2)
    z = np.fft.irfft(spec, N)[:n_pts]
    return z


def make_route(cycle, T_amb, duration=1200.0, dt=0.1, ds=0.05, seed=0):
    rng = np.random.default_rng(seed)
    if cycle == "mixed":
        parts, cur = [], 0.0
        for dur, vr, sp in WLTC_LIKE_PHASES:
            seg = _segment_profile(CYCLE_SPECS[sp], dur * duration / 1200.0, dt, rng, vr, cur)
            parts.append(seg)
            cur = seg[-1]
        v = np.concatenate(parts)
    else:
        v = _segment_profile(CYCLE_SPECS[cycle], duration, dt, rng)
    v[: int(5 / dt)] = 0.0                                  # start from rest
    tail = int(25 / dt)                                      # come to rest at the end
    v[-tail:] = v[-tail] * np.clip(np.linspace(1, -0.3, tail), 0, 1)
    v = np.maximum(_smooth(v, dt, 1.6), 0.0)
    n = int(duration / dt)
    v = np.pad(v, (0, max(0, n - len(v))))[:n]
    t = np.arange(n) * dt

    # desired lateral acceleration: curve events (bump-shaped), sign random
    a_lat = np.zeros(n)
    peak_rng = {"urban": (1.0, 3.2), "suburban": (0.8, 2.8), "highway": (0.4, 1.8),
                "hilly": (1.2, 3.5), "mixed": (0.6, 2.8)}[cycle]
    i = int(rng.uniform(10, 30) / dt)
    while i < n:
        dur = rng.uniform(3, 14)
        m = int(dur / dt)
        shape = np.sin(np.linspace(0, np.pi, m)) ** 2
        seg = rng.choice([-1, 1]) * rng.uniform(*peak_rng) * shape
        j = min(n, i + m)
        a_lat[i:j] += seg[: j - i]
        i = j + int(rng.exponential(25 if cycle != "highway" else 45) / dt)
    a_lat *= np.clip(v / 3.0, 0, 1)                          # no lateral accel at rest
    a_lat += _ou(n, dt, 1.5, 0.08, rng) * (v > 1)            # lane keeping wander

    # spatial profiles
    dist = np.sum(v) * dt * 1.08 + 200.0
    s = np.arange(0, dist, ds)
    corr = _smooth(rng.standard_normal(len(s)), ds, 350.0)
    corr /= corr.std() + 1e-12
    g_amp = 0.055 if cycle == "hilly" else 0.012
    grade = np.arctan(g_amp * np.tanh(corr))
    # roughness: patchwise ISO class
    probs = CYCLE_ROAD[cycle]
    classes, pk = list(probs.keys()), np.array(list(probs.values()))
    scale = np.zeros(len(s))
    k = 0
    while k < len(s):
        L = int(rng.uniform(200, 800) / ds)
        c = rng.choice(classes, p=pk / pk.sum())
        scale[k:k + L] = np.sqrt(ROAD_CLASS_Gd[c] / ROAD_CLASS_Gd["A"])
        k += L
    scale = _smooth(scale, ds, 20.0)
    z1 = _iso8608(len(s), ds, ROAD_CLASS_Gd["A"], rng)
    z2 = _iso8608(len(s), ds, ROAD_CLASS_Gd["A"], rng)
    rho = 0.6
    zL = z1 * scale
    zR = (rho * z1 + np.sqrt(1 - rho ** 2) * z2) * scale
    # discrete events: speed bumps (both tracks) and potholes (one track)
    n_bumps = int(dist / 1000 * (2 if cycle == "urban" else 0))
    for _ in range(n_bumps):
        x0 = rng.uniform(100, dist - 50)
        m = (s > x0) & (s < x0 + 3.7)
        bump = 0.04 * np.sin(np.pi * (s[m] - x0) / 3.7) ** 2
        zL[m] += bump
        zR[m] += bump
    for _ in range(int(dist / 1000 * 1.5)):
        x0 = rng.uniform(50, dist - 10)
        m = (s > x0) & (s < x0 + 0.6)
        side = zL if rng.random() < 0.5 else zR
        side[m] -= 0.025 * np.sin(np.pi * (s[m] - x0) / 0.6)

    # low-voltage loads: base + stochastic events + climate-dependent loads
    P = np.full(n, 0.0)
    P += 60 * _ou(n, dt, 20.0, 1.0, rng)
    for load, rate, dur in [(110, 1 / 600, (60, 400)), (80, 1 / 300, (20, 120)),
                            (45, 1 / 120, (5, 30)), (150, 1 / 400, (10, 60))]:
        i = int(rng.exponential(1 / rate) / dt)
        while i < n:
            m = int(rng.uniform(*dur) / dt)
            P[i:i + m] += load
            i += m + int(rng.exponential(1 / rate) / dt)
    if T_amb < 5:
        P[: int(600 / dt)] += 180                         # rear defrost
        P += 100                                          # seat heaters
    solar_base = {True: 450.0, False: 150.0}[T_amb > 28] if T_amb > 0 else 60.0
    solar = np.clip(solar_base * (1 + 0.3 * _ou(n, dt, 120.0, 1.0, rng)), 0, None)
    return Route(cycle, dt, t, v, a_lat, ds, s, grade, zL, zR, P, solar, T_amb)
