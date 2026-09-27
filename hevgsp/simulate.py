"""One scenario end to end: route -> powertrain/thermal -> suspension -> E/E -> sensors."""
import numpy as np
from .drivecycle import make_route
from .powertrain import simulate_powertrain
from .suspension import simulate_suspension
from .ee_sim import simulate_ee
from . import ee_arch as EA
from .graph import EE_NODES
from .sensors import measure, NOISE
from .graph import PHYS_NODES, names
from .faults import FAULTS, FAULT_IDS

DELTA_FACTOR = 10.0     # send-on-delta threshold = 10 x sensor noise std


def deltas():
    return np.array([DELTA_FACTOR * NOISE[n][0] for n, *_ in PHYS_NODES])


def run_scenario(cycle, T_amb, faults=(), seed=0, duration=1200.0, out_dt=0.1):
    route = make_route(cycle, T_amb, duration=duration + 1.0, seed=seed)
    pt, hi = simulate_powertrain(route, faults, seed=seed, out_dt=out_dt)
    sus = simulate_suspension(route, hi, faults, out_dt=out_dt)
    T = int(round(duration / out_dt))
    t = pt["t"][:T]
    sig = {**{k: v[:T] for k, v in pt.items()}, **{k: v[:T] for k, v in sus.items()}}
    Xp = np.column_stack([sig[n] for n, *_ in PHYS_NODES])
    units = {n: u for n, u, *_ in PHYS_NODES}
    pdus = EA.build_matrix(NOISE, units, DELTA_FACTOR)
    ee_out, ee_aux = simulate_ee(Xp, [n for n, *_ in PHYS_NODES], t, pdus, faults, seed=seed)
    Xe = np.column_stack([ee_out[n] for n, *_ in EE_NODES])
    X = np.hstack([Xp, Xe])
    # store ground truth on a power-of-two grid <= 1/64 of the CAN resolution (loss-free for
    # all practical purposes, and float32 mantissas compress well)
    q = np.array([NOISE[n][1] for n, *_ in PHYS_NODES] +
                 [0.1 if k == "bus" else 0.01 for _, _, k, _ in EE_NODES])
    g = 2.0 ** np.floor(np.log2(q / 64))
    X = (np.round(X / g) * g).astype(np.float32)
    Y = measure(X, t, faults, seed=seed)
    fid = np.zeros(T, np.int16)
    flevel = np.zeros(T, np.float32)
    for f in faults:
        L = np.asarray(f.level(t), float)
        fid[L > 0] = FAULT_IDS[f.name]
        flevel = np.maximum(flevel, (f.severity * L).astype(np.float32))
    aux = {k: sig[k].astype(np.float32) for k in
           ("engine_on", "s", "grade", "P_batt", "tq_eng_actual", "F_fric", "Q_hvac")}
    aux.update(ee_aux)
    meta = dict(cycle=cycle, T_amb=T_amb, seed=seed, duration=duration, dt=out_dt,
                faults=[dict(name=f.name, t0=f.t0, severity=f.severity,
                             layer=FAULTS[f.name]["layer"], component=FAULTS[f.name]["component"],
                             primary=FAULTS[f.name]["primary"], nodes=FAULTS[f.name]["nodes"])
                        for f in faults])
    return dict(t=t.astype(np.float32), X_clean=X, X_meas=Y, fault_id=fid,
                fault_level=flevel, aux=aux, meta=meta, v_ref=np.interp(t, route.t, route.v_ref))
