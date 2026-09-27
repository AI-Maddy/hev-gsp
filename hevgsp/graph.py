"""Multilayer graph: physical-sensor layer, E/E (ECU/bus) layer, interlayer
ownership edges. Edges in the physical layer are *structural*: an edge i-j
exists when the model equation governing one quantity contains the other
directly (see docs/MODEL.md). Each edge carries a coupling type."""
import numpy as np

# (name, unit, component, owner ECU)
PHYS_NODES = [
    ("v_veh", "m/s", "vehicle", "ESC"), ("a_long", "m/s^2", "imu", "ESC"),
    ("a_lat", "m/s^2", "imu", "ESC"), ("yaw_rate", "deg/s", "imu", "ESC"),
    ("steer_angle", "deg", "steering", "EPS"), ("pedal_acc", "%", "driver", "ZC_FRONT"),
    ("w_FL", "rad/s", "wheel_FL", "ESC"), ("w_FR", "rad/s", "wheel_FR", "ESC"),
    ("w_RL", "rad/s", "wheel_RL", "ESC"), ("w_RR", "rad/s", "wheel_RR", "ESC"),
    ("p_tire_FL", "bar", "tire_FL", "TPMS"), ("p_tire_FR", "bar", "tire_FR", "TPMS"),
    ("p_tire_RL", "bar", "tire_RL", "TPMS"), ("p_tire_RR", "bar", "tire_RR", "TPMS"),
    ("susp_travel_FL", "mm", "suspension_FL", "SCU"), ("susp_travel_FR", "mm", "suspension_FR", "SCU"),
    ("susp_travel_RL", "mm", "suspension_RL", "SCU"), ("susp_travel_RR", "mm", "suspension_RR", "SCU"),
    ("body_acc_FL", "m/s^2 rms", "suspension_FL", "SCU"), ("body_acc_FR", "m/s^2 rms", "suspension_FR", "SCU"),
    ("body_acc_RL", "m/s^2 rms", "suspension_RL", "SCU"), ("body_acc_RR", "m/s^2 rms", "suspension_RR", "SCU"),
    ("brake_pressure", "bar", "brakes", "ESC"), ("T_brake_F", "degC", "brakes_front", "ESC"),
    ("T_brake_R", "degC", "brakes_rear", "ESC"),
    ("n_eng", "rpm", "engine", "ECM"), ("tq_eng", "N m", "engine", "ECM"),
    ("fuel_rate", "g/s", "engine", "ECM"), ("T_cool_eng", "degC", "engine_cooling", "ECM"),
    ("T_cat", "degC", "exhaust", "ECM"),
    ("n_MG1", "rpm", "mg1", "PCU"), ("tq_MG1", "N m", "mg1", "PCU"),
    ("n_MG2", "rpm", "mg2", "PCU"), ("tq_MG2", "N m", "mg2", "PCU"),
    ("T_MG1", "degC", "mg1", "PCU"), ("T_MG2", "degC", "mg2", "PCU"), ("T_inv", "degC", "inverter", "PCU"),
    ("I_batt", "A", "battery", "BMS"), ("V_batt", "V", "battery", "BMS"),
    ("SOC", "%", "battery", "BMS"), ("T_batt", "degC", "battery", "BMS"),
    ("V_12", "V", "dcdc", "DCDC_ECU"), ("I_dcdc", "A", "dcdc", "DCDC_ECU"),
    ("T_cool_edu", "degC", "edu_cooling", "TMS"), ("T_rad_out", "degC", "engine_cooling", "TMS"),
    ("T_cabin", "degC", "cabin", "CCU"), ("P_ac", "W", "hvac", "CCU"),
]
EE_NODES = [  # (name, unit, kind, attachment); CP = AUTOSAR Classic, AP = Adaptive (CAPI)
    ("ECM", "ms", "cp_ecu", "CANFD_F"), ("PCU", "ms", "cp_ecu", "CANFD_F"), ("ESC", "ms", "cp_ecu", "CANFD_F"),
    ("EPS", "ms", "cp_ecu", "CANFD_F"), ("DCDC_ECU", "ms", "cp_ecu", "CANFD_F"), ("TMS", "ms", "cp_ecu", "CANFD_F"),
    ("BMS", "ms", "cp_ecu", "CANFD_R"), ("SCU", "ms", "cp_ecu", "CANFD_R"), ("TPMS", "ms", "cp_ecu", "CANFD_R"),
    ("CCU", "ms", "cp_ecu", "CANFD_R"),
    ("ZC_FRONT", "ms", "cp_zonal_gw", "CANFD_F"), ("ZC_REAR", "ms", "cp_zonal_gw", "CANFD_R"),
    ("CVC", "ms", "ap_machine", "ETH_BB"), ("TCU", "ms", "ap_machine", "ETH_BB"),
    ("CANFD_F", "%", "bus", None), ("CANFD_R", "%", "bus", None), ("ETH_BB", "%", "bus", None),
]

PHYS_EDGES = [
    # kinematic
    ("v_veh", "w_FL", "kin"), ("v_veh", "w_FR", "kin"), ("v_veh", "w_RL", "kin"), ("v_veh", "w_RR", "kin"),
    ("w_FL", "w_FR", "kin"), ("w_RL", "w_RR", "kin"), ("w_FL", "w_RL", "kin"), ("w_FR", "w_RR", "kin"),
    ("v_veh", "a_long", "kin"), ("yaw_rate", "a_lat", "kin"), ("v_veh", "a_lat", "kin"),
    ("yaw_rate", "steer_angle", "kin"), ("a_lat", "steer_angle", "kin"),
    ("n_MG2", "w_FL", "kin"), ("n_MG2", "w_FR", "kin"),
    ("yaw_rate", "w_FL", "kin"), ("yaw_rate", "w_FR", "kin"), ("yaw_rate", "w_RL", "kin"), ("yaw_rate", "w_RR", "kin"),
    ("n_eng", "n_MG1", "kin"), ("n_MG1", "n_MG2", "kin"), ("n_eng", "n_MG2", "kin"),
    # mechanical
    ("tq_eng", "tq_MG1", "mech"), ("tq_eng", "n_eng", "mech"), ("tq_MG2", "a_long", "mech"),
    ("tq_eng", "a_long", "mech"), ("brake_pressure", "a_long", "mech"),
    ("p_tire_FL", "w_FL", "mech"), ("p_tire_FR", "w_FR", "mech"),
    ("p_tire_RL", "w_RL", "mech"), ("p_tire_RR", "w_RR", "mech"),
    ("susp_travel_FL", "body_acc_FL", "mech"), ("susp_travel_FR", "body_acc_FR", "mech"),
    ("susp_travel_RL", "body_acc_RL", "mech"), ("susp_travel_RR", "body_acc_RR", "mech"),
    ("susp_travel_FL", "susp_travel_FR", "mech"), ("susp_travel_RL", "susp_travel_RR", "mech"),
    ("susp_travel_FL", "susp_travel_RL", "mech"), ("susp_travel_FR", "susp_travel_RR", "mech"),
    ("body_acc_FL", "body_acc_FR", "mech"), ("body_acc_RL", "body_acc_RR", "mech"),
    ("body_acc_FL", "body_acc_RL", "mech"), ("body_acc_FR", "body_acc_RR", "mech"),
    ("a_long", "susp_travel_FL", "mech"), ("a_long", "susp_travel_FR", "mech"),
    ("a_long", "susp_travel_RL", "mech"), ("a_long", "susp_travel_RR", "mech"),
    ("a_lat", "susp_travel_FL", "mech"), ("a_lat", "susp_travel_FR", "mech"),
    ("a_lat", "susp_travel_RL", "mech"), ("a_lat", "susp_travel_RR", "mech"),
    ("v_veh", "body_acc_FL", "mech"), ("v_veh", "body_acc_FR", "mech"),
    ("v_veh", "body_acc_RL", "mech"), ("v_veh", "body_acc_RR", "mech"),
    ("fuel_rate", "tq_eng", "mech"), ("fuel_rate", "n_eng", "mech"),
    # control (HCU / driver / brake blending / charge sustaining)
    ("pedal_acc", "tq_MG2", "ctrl"), ("pedal_acc", "tq_eng", "ctrl"), ("pedal_acc", "a_long", "ctrl"),
    ("brake_pressure", "tq_MG2", "ctrl"), ("SOC", "tq_eng", "ctrl"),
    # electrical
    ("I_batt", "tq_MG1", "elec"), ("I_batt", "tq_MG2", "elec"), ("I_batt", "V_batt", "elec"),
    ("V_batt", "SOC", "elec"), ("I_batt", "SOC", "elec"), ("I_batt", "I_dcdc", "elec"),
    ("I_batt", "P_ac", "elec"), ("V_12", "I_dcdc", "elec"),
    # thermal
    ("T_batt", "I_batt", "therm"), ("T_batt", "T_cabin", "therm"), ("T_batt", "V_batt", "elec"),
    ("T_MG1", "tq_MG1", "therm"), ("T_MG2", "tq_MG2", "therm"),
    ("T_MG1", "T_cool_edu", "therm"), ("T_MG2", "T_cool_edu", "therm"), ("T_inv", "T_cool_edu", "therm"),
    ("T_inv", "I_batt", "therm"), ("T_cool_edu", "v_veh", "therm"),
    ("T_cool_eng", "T_rad_out", "therm"), ("T_rad_out", "v_veh", "therm"),
    ("T_cool_eng", "fuel_rate", "therm"), ("T_cool_eng", "T_cabin", "therm"),
    ("T_cat", "fuel_rate", "therm"), ("T_cat", "n_eng", "therm"),
    ("T_cabin", "P_ac", "therm"),
    ("T_brake_F", "brake_pressure", "therm"), ("T_brake_R", "brake_pressure", "therm"),
    ("T_brake_F", "v_veh", "therm"), ("T_brake_R", "v_veh", "therm"),
    ("p_tire_FL", "p_tire_FR", "therm"), ("p_tire_RL", "p_tire_RR", "therm"),
    ("p_tire_FL", "p_tire_RL", "therm"), ("p_tire_FR", "p_tire_RR", "therm"),
    ("p_tire_FL", "v_veh", "therm"), ("p_tire_FR", "v_veh", "therm"),
    ("p_tire_RL", "v_veh", "therm"), ("p_tire_RR", "v_veh", "therm"),
]
EE_EDGES = [(e, bus, "canfd") for e, _, kind, bus in EE_NODES if kind in ("cp_ecu", "cp_zonal_gw")] + [
    ("ZC_FRONT", "ETH_BB", "eth"), ("ZC_REAR", "ETH_BB", "eth"), ("CVC", "ETH_BB", "eth"), ("TCU", "ETH_BB", "eth"),
    # ara::com (SOME/IP) service provider-consumer relations
    ("CVC", "ZC_FRONT", "svc"), ("CVC", "ZC_REAR", "svc"), ("TCU", "ZC_FRONT", "svc"), ("TCU", "ZC_REAR", "svc")]
INTER_EDGES = [(n, owner, "own") for n, _, _, owner in PHYS_NODES]

EDGE_TYPES = ["kin", "mech", "ctrl", "elec", "therm", "canfd", "eth", "svc", "own"]


def node_table():
    rows = []
    for n, u, comp, owner in PHYS_NODES:
        rows.append(dict(name=n, unit=u, layer="physical", component=comp, owner=owner))
    for n, u, kind, bus in EE_NODES:
        rows.append(dict(name=n, unit=u, layer="ee", component=kind, owner=bus or ""))
    return rows


def names():
    return [r["name"] for r in node_table()]


def build(inter_weight=1.0):
    nm = names()
    idx = {n: i for i, n in enumerate(nm)}
    N = len(nm)
    A = np.zeros((N, N))
    T = np.full((N, N), "", dtype=object)
    edges = []
    for (a, b, ty), w in [(e, 1.0) for e in PHYS_EDGES + EE_EDGES] + \
                         [(e, inter_weight) for e in INTER_EDGES]:
        i, j = idx[a], idx[b]
        assert i != j and A[i, j] == 0, (a, b)
        A[i, j] = A[j, i] = w
        T[i, j] = T[j, i] = ty
        edges.append((i, j, ty))
    nP = len(PHYS_NODES)
    layer = np.array([0] * nP + [1] * len(EE_NODES))
    return dict(names=nm, index=idx, A=A, types=T, edges=edges, layer=layer, n_phys=nP)


def laplacian(A, normalized=False):
    d = A.sum(1)
    L = np.diag(d) - A
    if normalized:
        di = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
        L = di[:, None] * L * di[None, :]
    return L


def signed_laplacian(W):
    """Signed Laplacian L = D_|W| - W (PSD); x^T L x = sum |w_ij| (x_i - sgn(w_ij) x_j)^2."""
    return np.diag(np.abs(W).sum(1)) - W
