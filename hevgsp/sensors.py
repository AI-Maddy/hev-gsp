"""Measurement model: additive Gaussian noise + quantization per channel,
multiplicative jitter for network latencies, and sensor-level faults."""
import numpy as np
from .graph import names, PHYS_NODES, EE_NODES
from .faults import lvl
from .ee_arch import RANGES

# name: (noise std, quantization step)
NOISE = {
    "v_veh": (0.03, 0.01), "a_long": (0.03, 0.01), "a_lat": (0.03, 0.01), "yaw_rate": (0.15, 0.05),
    "steer_angle": (0.5, 0.1), "pedal_acc": (0.3, 0.4),
    "w_FL": (0.08, 0.03), "w_FR": (0.08, 0.03), "w_RL": (0.08, 0.03), "w_RR": (0.08, 0.03),
    "p_tire_FL": (0.008, 0.014), "p_tire_FR": (0.008, 0.014), "p_tire_RL": (0.008, 0.014), "p_tire_RR": (0.008, 0.014),
    "susp_travel_FL": (0.5, 0.1), "susp_travel_FR": (0.5, 0.1), "susp_travel_RL": (0.5, 0.1), "susp_travel_RR": (0.5, 0.1),
    "body_acc_FL": (0.02, 0.01), "body_acc_FR": (0.02, 0.01), "body_acc_RL": (0.02, 0.01), "body_acc_RR": (0.02, 0.01),
    "brake_pressure": (0.1, 0.1), "T_brake_F": (2.0, 1.0), "T_brake_R": (2.0, 1.0),
    "n_eng": (5.0, 0.25), "tq_eng": (2.0, 0.5), "fuel_rate": (0.02, 0.01), "T_cool_eng": (0.3, 1.0), "T_cat": (4.0, 5.0),
    "n_MG1": (5.0, 1.0), "tq_MG1": (1.0, 0.25), "n_MG2": (5.0, 1.0), "tq_MG2": (1.0, 0.25),
    "T_MG1": (0.4, 1.0), "T_MG2": (0.4, 1.0), "T_inv": (0.4, 1.0),
    "I_batt": (0.5, 0.1), "V_batt": (0.3, 0.1), "SOC": (0.1, 0.1), "T_batt": (0.3, 0.5),
    "V_12": (0.02, 0.01), "I_dcdc": (0.3, 0.1),
    "T_cool_edu": (0.3, 0.5), "T_rad_out": (0.3, 1.0), "T_cabin": (0.2, 0.5), "P_ac": (20.0, 10.0),
}
EE_JITTER = 0.01       # relative std (timestamp resolution / tracing overhead)
BUS_NOISE = 0.2        # % absolute for bus utilization


def measure(X, t, faults=(), seed=0):
    rng = np.random.default_rng(seed + 23)
    nm = names()
    Y = X.copy()
    nP = len(PHYS_NODES)
    for j, n in enumerate(nm[:nP]):
        s, q = NOISE[n]
        y = X[:, j] + s * rng.standard_normal(len(t))
        lo, hi = RANGES[n]                       # CAN signal range (CompuMethod limits)
        off = {"n_eng": 0.0, "T_cool_eng": -40.0}.get(n, lo)    # CompuMethod offset (J1979 where defined)
        Y[:, j] = np.clip(np.round((y - off) / q) * q + off, lo, hi)
    for j, (n, u, kind, bus) in enumerate(EE_NODES):
        k = nP + j
        if kind == "bus":
            y = X[:, k] + BUS_NOISE * rng.standard_normal(len(t))
            Y[:, k] = np.round(y * 128) / 128                  # ~0.008 % resolution
        else:
            y = X[:, k] * (1 + EE_JITTER * rng.standard_normal(len(t)))
            Y[:, k] = np.round(y * 1024) / 1024                # ~1 us timestamp resolution
    idx = {n: i for i, n in enumerate(nm)}
    for f in faults:
        if f.name == "sens_bias_yaw":
            Y[:, idx["yaw_rate"]] += 3.0 * f.severity * f.level(t)
        elif f.name == "sens_stuck_Tbatt":
            j = idx["T_batt"]
            on = t >= f.t0
            k0 = np.argmax(on)
            Y[on, j] = np.round((Y[k0, j] - 6.0 * f.severity) / 0.5) * 0.5
        elif f.name == "sens_noise_wFR":
            j = idx["w_FR"]
            on = t >= f.t0
            Y[on, j] += 0.8 * f.severity * rng.standard_normal(on.sum())
            # intermittent dropouts in bursts
            drop = np.zeros(len(t), bool)
            i = int(np.argmax(on))
            while i < len(t):
                i += int(rng.exponential(80 / f.severity))
                L = int(rng.integers(2, 12))
                drop[i:i + L] = True
                i += L
            Y[drop & on, j] = 0.0
    return Y.astype(np.float32)
