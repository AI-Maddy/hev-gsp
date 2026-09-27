"""Fault catalogue. Physical faults act inside the model equations, so their
effects propagate through the coupled dynamics; sensor faults corrupt one
measurement channel only; E/E faults act on the network layer."""
from dataclasses import dataclass
import numpy as np

FAULTS = {
    # name: layer, component, primary node, affected-component nodes, ramp (s), description
    "bat_resistance":   dict(layer="physical", component="battery", primary="V_batt",
                             nodes=["I_batt", "V_batt", "SOC", "T_batt"], ramp=60,
                             desc="Cell internal-resistance growth (R0 x1.75 / x2.5)"),
    "edu_pump":         dict(layer="physical", component="edu_cooling", primary="T_cool_edu",
                             nodes=["T_cool_edu", "T_MG1", "T_MG2", "T_inv"], ramp=30,
                             desc="Electric-drive coolant pump flow loss (-37% / -75%)"),
    "mg2_winding":      dict(layer="physical", component="mg2", primary="T_MG2",
                             nodes=["T_MG2", "tq_MG2", "n_MG2"], ramp=60,
                             desc="MG2 stator inter-turn short: circulating loss ~ speed^2 (250/500 W at 1000 rad/s), copper loss x2 / x3"),
    "misfire":          dict(layer="physical", component="engine", primary="tq_MG1",
                             nodes=["tq_MG1", "tq_eng", "n_eng", "fuel_rate", "T_cat", "T_cool_eng"], ramp=5,
                             desc="Cylinder misfire, 12% / 25% torque loss, catalyst heating"),
    "thermostat_stuck": dict(layer="physical", component="engine_cooling", primary="T_cool_eng",
                             nodes=["T_cool_eng", "T_rad_out"], ramp=10,
                             desc="Thermostat stuck 60% / 100% open"),
    "tire_leak_FL":     dict(layer="physical", component="tire_FL", primary="p_tire_FL",
                             nodes=["p_tire_FL", "w_FL", "susp_travel_FL", "body_acc_FL"], ramp=300,
                             desc="Slow leak front-left, -22% / -45% cold pressure (tau 300 s)"),
    "damper_RR":        dict(layer="physical", component="damper_RR", primary="body_acc_RR",
                             nodes=["body_acc_RR", "susp_travel_RR"], ramp=20,
                             desc="Rear-right damper oil loss, damping -40% / -80%"),
    "brake_drag_F":     dict(layer="physical", component="brakes_front", primary="T_brake_F",
                             nodes=["T_brake_F"], ramp=10,
                             desc="Front caliper drag, 125 N / 250 N residual force"),
    "dcdc_degradation": dict(layer="physical", component="dcdc", primary="V_12",
                             nodes=["V_12", "I_dcdc"], ramp=30,
                             desc="DC-DC output sag 0.4 / 0.8 V, efficiency loss, ripple"),
    "sens_bias_yaw":    dict(layer="sensor", component="imu", primary="yaw_rate",
                             nodes=["yaw_rate"], ramp=120,
                             desc="Yaw-rate sensor bias drift to 1.5 / 3 deg/s"),
    "sens_stuck_Tbatt": dict(layer="sensor", component="bms_sensor", primary="T_batt",
                             nodes=["T_batt"], ramp=0,
                             desc="Battery temperature sensor stuck at last value (+/- offset)"),
    "sens_noise_wFR":   dict(layer="sensor", component="wss_FR", primary="w_FR",
                             nodes=["w_FR"], ramp=0,
                             desc="Wheel-speed sensor FR intermittent dropouts and noise"),
    "ee_babbling_ESC":  dict(layer="ee", component="ESC", primary="ESC",
                             nodes=["ESC", "CANFD_F"], ramp=0,
                             desc="ESC babbling idiot: spurious high-priority CAN FD frames, 1750 / 3500 fps"),
    "ee_zonal_overload": dict(layer="ee", component="ZC_FRONT", primary="ZC_FRONT",
                             nodes=["ZC_FRONT", "CVC", "ETH_BB"], ramp=20,
                             desc="Remote DoIP (ISO 13400) UDS request flood via TCU saturating the front zonal gateway"),
    "ee_bms_overrun":   dict(layer="ee", component="BMS", primary="BMS",
                             nodes=["BMS"], ramp=10,
                             desc="BMS OS task overrun, CPU load +12% / +25%"),
}
FAULT_IDS = {name: i + 1 for i, name in enumerate(FAULTS)}   # 0 = nominal


@dataclass
class Fault:
    name: str
    t0: float
    severity: float           # 0.5 (moderate) or 1.0 (severe)

    def level(self, t):
        """Activation profile in [0, 1] (scalar or array t)."""
        ramp = FAULTS[self.name]["ramp"]
        t = np.asarray(t, dtype=float)
        if self.name == "tire_leak_FL":
            x = 1 - np.exp(-np.clip(t - self.t0, 0, None) / ramp)
        elif ramp == 0:
            x = (t >= self.t0).astype(float)
        else:
            x = np.clip((t - self.t0) / ramp, 0, 1)
        return x if x.ndim else float(x)


def lvl(faults, name, t):
    """Severity-scaled level of a named fault (0 if absent)."""
    for f in faults:
        if f.name == name:
            return f.severity * f.level(t)
    return 0.0
