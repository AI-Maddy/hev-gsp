"""7-DOF full-vehicle ride model (heave, pitch, roll + 4 unsprung masses).

Inputs: road heights under each tire (ISO 8608 tracks, rear delayed by the
wheelbase in space), pitch and roll inertial moments from the longitudinal /
lateral accelerations. Tire stiffness is scheduled from tire pressure and the
damper-fault level, so the model is piecewise LTI (re-discretized per block).
Outputs per corner: suspension travel (mm, + = extension) and body vertical
acceleration; these are reduced to the output rate as block mean (travel)
and block RMS (acceleration).
"""
import numpy as np
from scipy.signal import cont2discrete, dlsim
from .params import Vehicle
from .faults import lvl

CORNERS = ("FL", "FR", "RL", "RR")


def _ss(V, kt, cmult):
    s, ch = V.sus, V.ch
    xs = np.array([ch.a, ch.a, -ch.b, -ch.b])
    ys = np.array([ch.track / 2, -ch.track / 2, ch.track / 2, -ch.track / 2])
    ks = np.array([s.ks_f, s.ks_f, s.ks_r, s.ks_r])
    cs = np.array([s.cs_f, s.cs_f, s.cs_r, s.cs_r]) * cmult
    mu = np.array([s.mu_f, s.mu_f, s.mu_r, s.mu_r])
    karb = np.array([s.k_arb_f, s.k_arb_f, s.k_arb_r, s.k_arb_r])
    # generalized coords q = [zs, th, ph, zu1..4]; corner body disp zb_i = zs + x_i th + y_i ph
    Tb = np.zeros((4, 7))
    Tb[:, 0] = 1; Tb[:, 1] = xs; Tb[:, 2] = ys
    Td = Tb.copy(); Td[:, 3:] = -np.eye(4)                # deflection d_i = zb_i - zu_i
    K_c = np.diag(ks)
    # anti-roll bars couple left/right deflections on each axle
    Karb = np.zeros((4, 4))
    for (l, r) in ((0, 1), (2, 3)):
        Karb[l, l] += karb[l]; Karb[r, r] += karb[r]
        Karb[l, r] -= karb[l]; Karb[r, l] -= karb[r]
    K = Td.T @ (K_c + Karb) @ Td
    C = Td.T @ np.diag(cs) @ Td
    K[3:, 3:] += np.diag(kt)
    M = np.diag([s.ms, s.Iy, s.Ix, *mu])
    Mi = np.linalg.inv(M)
    # inputs u = [zr1..4, M_pitch, M_roll]
    Bq = np.zeros((7, 6))
    Bq[3:, :4] = np.diag(kt)
    Bq[1, 4] = 1.0; Bq[2, 5] = 1.0
    A = np.block([[np.zeros((7, 7)), np.eye(7)], [-Mi @ K, -Mi @ C]])
    B = np.vstack([np.zeros((7, 6)), Mi @ Bq])
    # outputs: deflection (4), body corner acceleration (4)
    Cy = np.vstack([np.hstack([Td, np.zeros((4, 7))]), Tb @ np.hstack([-Mi @ K, -Mi @ C])])
    Dy = np.vstack([np.zeros((4, 6)), Tb @ Mi @ Bq])
    return A, B, Cy, Dy


def simulate_suspension(route, hi, faults=(), veh=None, out_dt=0.1, block_s=1.0):
    V = veh or Vehicle()
    fs = V.sus.fs
    dt = 1.0 / fs
    t_end = hi["t"][-1]
    t = np.arange(0, t_end, dt)
    s_pos = np.interp(t, hi["t"], hi["s"])
    ax = np.interp(t, hi["t"], hi["a_x"])
    ay = np.interp(t, hi["t"], hi["a_y"])
    s_rear = np.maximum(s_pos - V.ch.wheelbase, 0.0)
    zr = np.stack([np.interp(s_pos, route.s, route.zL), np.interp(s_pos, route.s, route.zR),
                   np.interp(s_rear, route.s, route.zL), np.interp(s_rear, route.s, route.zR)], 1)
    Mp = V.sus.ms * ax * V.ch.h_cg
    Mr = V.sus.ms * ay * V.ch.h_cg
    u = np.column_stack([zr, Mp, Mr])

    nb = int(np.ceil(len(t) / (block_s * fs)))
    x = np.zeros(14)
    ys = []
    cache = {}
    for b in range(nb):
        i0 = int(b * block_s * fs)
        i1 = min(len(t), int((b + 1) * block_s * fs))
        tm = t[i0]                     # level at block start: effect never precedes the label
        kt = np.array([np.interp(tm, hi["t"], hi["k_t"][:, j]) for j in range(4)])
        cm = np.ones(4)
        cm[3] = 1 - 0.8 * lvl(faults, "damper_RR", tm)
        key = tuple(np.round(kt, -2)) + tuple(np.round(cm, 3))
        if key not in cache:
            A, B, C, D = _ss(V, kt, cm)
            cache[key] = cont2discrete((A, B, C, D), dt, method="zoh")[:4]
        Ad, Bd, Cd, Dd = cache[key]
        ub = u[i0:i1]
        if len(ub) < 2:
            ub = np.vstack([ub, ub])
        _, yb, xb = dlsim((Ad, Bd, Cd, Dd, dt), ub, x0=x)
        yb, xb = yb[: i1 - i0], xb[: i1 - i0]
        x = Ad @ xb[-1] + Bd @ u[i1 - 1]
        ys.append(yb)
    y = np.vstack(ys)
    k = int(round(out_dt * fs))
    m = len(t) // k
    y = y[: m * k].reshape(m, k, 8)
    travel = y[:, :, :4].mean(1) * 1000.0                 # mm
    acc_rms = np.sqrt((y[:, :, 4:] ** 2).mean(1))          # m/s^2
    out = {}
    for j, c in enumerate(CORNERS):
        out["susp_travel_" + c] = travel[:, j]
        out["body_acc_" + c] = acc_rms[:, j]
    return out
