"""Zonal, service-oriented E/E architecture.

Classic Platform (CP) endpoint ECUs exchange COM signals packed into
I-PDUs carried on CAN FD (ISO 11898-1:2015, 500 kbit/s arbitration / 2 Mbit/s
data). Two CP zonal controllers route I-PDUs (PduR / COM signal gateway) and
translate them into SOME/IP events on a 100BASE-T1 backbone. The central
vehicle computer (CVC) and telematics unit (TCU) are Adaptive Platform
machines (the platform CAPI implements): the hybrid energy-management function
runs on the CVC as an Adaptive Application consuming ara::com events and
publishing torque/speed requests back to the zones.

This module holds the static description: ECUs, buses, the COM/I-PDU
communication matrix (with CompuMethod scaling), and the SOME/IP deployment.
"""
from dataclasses import dataclass, field
import math
import numpy as np

CP_ECUS = ["ECM", "PCU", "ESC", "EPS", "DCDC_ECU", "TMS", "BMS", "SCU", "TPMS", "CCU"]
ZONES = {"ZC_FRONT": "CANFD_F", "ZC_REAR": "CANFD_R"}
BUS_OF = {"ECM": "CANFD_F", "PCU": "CANFD_F", "ESC": "CANFD_F", "EPS": "CANFD_F",
          "DCDC_ECU": "CANFD_F", "TMS": "CANFD_F", "BMS": "CANFD_R", "SCU": "CANFD_R",
          "TPMS": "CANFD_R", "CCU": "CANFD_R"}
ZONE_OF = {e: ("ZC_FRONT" if b == "CANFD_F" else "ZC_REAR") for e, b in BUS_OF.items()}
CAN_BUSES = {"CANFD_F": dict(nominal=500e3, data=2e6), "CANFD_R": dict(nominal=500e3, data=2e6)}
ETH = dict(name="ETH_BB", rate=100e6)          # 100BASE-T1 backbone (IEEE 802.3bw)

# ECU OS/COM timing: Com_MainFunctionTx period (ms), COM task WCET C (ms),
# base CPU load, CPU load per transmitted frame/s. Fixed-priority preemptive OS.
ECU_TIMING = {
    "ECM": (5, 0.35, 0.45, 6e-5), "PCU": (5, 0.30, 0.50, 6e-5), "ESC": (5, 0.25, 0.45, 5e-5),
    "EPS": (5, 0.30, 0.35, 6e-5), "DCDC_ECU": (10, 0.40, 0.25, 8e-5), "TMS": (10, 0.40, 0.25, 8e-5),
    "BMS": (10, 0.45, 0.35, 8e-5), "SCU": (5, 0.30, 0.35, 6e-5), "TPMS": (20, 0.60, 0.15, 1e-4),
    "CCU": (20, 0.60, 0.25, 1e-4), "ZC_FRONT": (5, 0.10, 0.30, 0), "ZC_REAR": (5, 0.10, 0.25, 0),
}

# physical signal ranges (for CompuMethod offset/bit-length) and resolution
RANGES = {
    "v_veh": (0, 70), "a_long": (-15, 15), "a_lat": (-15, 15), "yaw_rate": (-150, 150),
    "steer_angle": (-800, 800), "pedal_acc": (0, 100),
    "w_FL": (0, 250), "w_FR": (0, 250), "w_RL": (0, 250), "w_RR": (0, 250),
    "p_tire_FL": (0, 5), "p_tire_FR": (0, 5), "p_tire_RL": (0, 5), "p_tire_RR": (0, 5),
    "susp_travel_FL": (-120, 120), "susp_travel_FR": (-120, 120), "susp_travel_RL": (-120, 120), "susp_travel_RR": (-120, 120),
    "body_acc_FL": (0, 20), "body_acc_FR": (0, 20), "body_acc_RL": (0, 20), "body_acc_RR": (0, 20),
    "brake_pressure": (0, 250), "T_brake_F": (-40, 800), "T_brake_R": (-40, 800),
    "n_eng": (0, 8000), "tq_eng": (-50, 250), "fuel_rate": (0, 20), "T_cool_eng": (-40, 215), "T_cat": (-40, 1200),
    "n_MG1": (-15000, 15000), "tq_MG1": (-200, 200), "n_MG2": (-15000, 15000), "tq_MG2": (-400, 400),
    "T_MG1": (-40, 215), "T_MG2": (-40, 215), "T_inv": (-40, 215),
    "I_batt": (-400, 400), "V_batt": (0, 500), "SOC": (0, 100), "T_batt": (-40, 100),
    "V_12": (0, 20), "I_dcdc": (0, 250), "T_cool_edu": (-40, 215), "T_rad_out": (-40, 215),
    "T_cabin": (-40, 100), "P_ac": (0, 10000),
}
# Where a signal has an SAE J1979 (OBD-II) mode-01 PID, the CAN resolution follows its scaling.
J1979 = {"n_eng": ("PID 0x0C", 0.25, 0.0), "T_cool_eng": ("PID 0x05", 1.0, -40.0)}

CANFD_DLC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]


@dataclass
class Signal:
    name: str
    factor: float
    offset: float
    length: int
    start: int = 0
    unit: str = ""
    trigger: bool = False          # TRIGGERED_ON_CHANGE
    mask_bits: int = 0             # MASKED_NEW_DIFFERS_MASKED_OLD mask: low bits cleared
    graph_node: bool = True


@dataclass
class IPdu:
    name: str
    sender: str
    bus: str
    can_id: int
    period_ms: float
    mode: str                      # PERIODIC or MIXED (ComTxModeMode)
    mdt_ms: float                  # ComMinimumDelayTime
    payload: int = 8               # bytes (CAN FD DLC-valid)
    signals: list = field(default_factory=list)
    e2e: str = ""                  # E2E profile (AUTOSAR PRS_E2EProtocol), e.g. "P11", "P05"
    routed: bool = False           # routed by zone gateway to SOME/IP
    timeout_ms: float = 0.0        # receiver ComTimeout (deadline monitoring)


def _signal(name, q, unit, trigger, delta):
    lo, hi = RANGES[name]
    if name in J1979:
        _, q, off = J1979[name]
    else:
        off = lo
    L = int(math.ceil(math.log2((hi - off) / q + 1)))
    mask_bits = int(max(0, math.floor(math.log2(max(delta / q, 1.0))))) if trigger else 0
    return Signal(name, q, off, L, unit=unit, trigger=trigger, mask_bits=mask_bits)


def _fit_payload(bits):
    need = int(math.ceil(bits / 8))
    return next(d for d in CANFD_DLC if d >= need)


def build_matrix(noise, units, delta_factor=10.0, seed=2026):
    """Communication matrix: graph-signal PDUs + command PDUs + background PDUs."""
    rng = np.random.default_rng(seed)
    # signal PDUs: (pdu name, sender, period ms, mode, mdt ms, e2e, [signals])
    spec = [
        ("ECM_EngSts", "ECM", 10, "MIXED", 2, "P11", ["n_eng", "tq_eng", "fuel_rate"]),
        ("ECM_EngThm", "ECM", 100, "PERIODIC", 0, "", ["T_cool_eng", "T_cat"]),
        ("PCU_MgSts", "PCU", 10, "MIXED", 2, "P11", ["n_MG1", "tq_MG1", "n_MG2", "tq_MG2"]),
        ("PCU_MgThm", "PCU", 100, "PERIODIC", 0, "", ["T_MG1", "T_MG2", "T_inv"]),
        ("ESC_VehDyn", "ESC", 10, "PERIODIC", 0, "P11", ["v_veh", "a_long", "a_lat", "yaw_rate"]),
        ("ESC_WhlSpd", "ESC", 10, "PERIODIC", 0, "P11", ["w_FL", "w_FR", "w_RL", "w_RR"]),
        ("ESC_Brk", "ESC", 20, "MIXED", 5, "P11", ["brake_pressure", "T_brake_F", "T_brake_R"]),
        ("EPS_Sts", "EPS", 10, "PERIODIC", 0, "P11", ["steer_angle"]),
        ("DCDC_Sts", "DCDC_ECU", 100, "MIXED", 10, "", ["V_12", "I_dcdc"]),
        ("TMS_Sts", "TMS", 100, "PERIODIC", 0, "", ["T_cool_edu", "T_rad_out"]),
        ("BMS_PackSts", "BMS", 20, "MIXED", 5, "P05", ["I_batt", "V_batt", "SOC"]),
        ("BMS_Thm", "BMS", 100, "PERIODIC", 0, "", ["T_batt"]),
        ("SCU_Travel", "SCU", 10, "PERIODIC", 0, "", ["susp_travel_FL", "susp_travel_FR", "susp_travel_RL", "susp_travel_RR"]),
        ("SCU_BodyAcc", "SCU", 20, "PERIODIC", 0, "", ["body_acc_FL", "body_acc_FR", "body_acc_RL", "body_acc_RR"]),
        ("TPMS_Press", "TPMS", 1000, "MIXED", 100, "", ["p_tire_FL", "p_tire_FR", "p_tire_RL", "p_tire_RR"]),
        ("CCU_Sts", "CCU", 200, "MIXED", 50, "", ["T_cabin", "P_ac"]),
    ]
    # CAN identifiers: lower = higher arbitration priority (safety / dynamics first)
    pri = {"ESC": 0x080, "ZC_FRONT": 0x0C0, "EPS": 0x100, "PCU": 0x140, "ECM": 0x180, "ZC_REAR": 0x1C0,
           "BMS": 0x200, "SCU": 0x240, "DCDC_ECU": 0x300, "TMS": 0x340, "TPMS": 0x400, "CCU": 0x440}
    used = {}
    pdus = []
    for name, snd, per, mode, mdt, e2e, sigs in spec:
        cid = pri[snd] + used.get(snd, 0)
        used[snd] = used.get(snd, 0) + 2
        S = []
        hdr = 16 if e2e == "P11" else 24 if e2e == "P05" else 0     # E2E header bits (CRC + counter)
        start = hdr
        for s in sigs:
            sig = _signal(s, noise[s][1], units[s], mode == "MIXED", delta_factor * noise[s][0])
            sig.start = start
            start += sig.length
            S.append(sig)
        pdus.append(IPdu(name, snd, BUS_OF[snd], cid, per, mode, mdt, _fit_payload(start),
                         S, e2e, True, 2.5 * per))
    # command PDUs from zones (routed from CVC ara::com methods/events)
    for name, zc, per, payload, e2e in [("ZCF_TqReqEng", "ZC_FRONT", 10, 8, "P11"),
                                        ("ZCF_TqReqMg", "ZC_FRONT", 10, 8, "P11"),
                                        ("ZCF_ThmReq", "ZC_FRONT", 100, 8, ""),
                                        ("ZCR_BmsReq", "ZC_REAR", 20, 8, "P05"),
                                        ("ZCR_ClimReq", "ZC_REAR", 200, 8, "")]:
        cid = pri[zc] + used.get(zc, 0)
        used[zc] = used.get(zc, 0) + 2
        pdus.append(IPdu(name, zc, ZONES[zc], cid, per, "PERIODIC", 0, payload, [], e2e, False, 2.5 * per))
    # background PDUs (functions not modeled as graph nodes: diagnostics status,
    # NM PDUs (CanNm), internal states). Deterministic from seed.
    for ecu in CP_ECUS:
        nbg = {"ECM": 7, "PCU": 6, "ESC": 8, "EPS": 4, "DCDC_ECU": 3, "TMS": 4,
               "BMS": 11, "SCU": 7, "TPMS": 3, "CCU": 8}[ecu]
        for k in range(nbg):
            per = float(rng.choice([10, 20, 50, 100, 100, 200, 500]))
            payload = int(rng.choice([8, 8, 16, 24, 32, 64]))
            cid = 0x480 + len(pdus) * 2 if per >= 100 else pri[ecu] + 0x10 + used.get(ecu, 0)
            used[ecu] = used.get(ecu, 0) + 2
            pdus.append(IPdu(f"{ecu}_Bg{k}", ecu, BUS_OF[ecu], cid, per, "PERIODIC", 0, payload,
                             [], "", False, 0.0))
        pdus.append(IPdu(f"{ecu}_NM", ecu, BUS_OF[ecu], 0x700 + CP_ECUS.index(ecu), 1000, "PERIODIC", 0, 8,
                         [], "", False, 0.0))
    ids = [p.can_id for p in pdus]
    assert len(ids) == len(set(ids)), "duplicate CAN IDs"
    return pdus


STUFF_P = 1.0 / 30.0       # probability of a dynamic stuff bit per stuffable bit (random payload)


def canfd_bits(payload, dyn=None):
    """CAN FD base-format frame (ISO 11898-1:2015). Returns (bits at nominal rate, bits at data rate).
    Arbitration phase SOF..BRS = 17 bits; data phase ESI, DLC, data, stuff-bit count (4) and CRC-17
    (payload <= 16 B) or CRC-21 with fixed stuff bits (one before the stuff count plus one every 4 bits);
    tail CRC delimiter, ACK slot, ACK delimiter, EOF (7) and IFS (3) = 13 bits. `dyn` = number of dynamic
    stuff bits in SOF..data; None = worst case."""
    arb = 17
    crc = 17 if payload <= 16 else 21
    stuffable = arb + 1 + 4 + 8 * payload
    worst = (stuffable - 1) // 4
    dyn = worst if dyn is None else min(dyn, worst)
    dyn_arb = int(round(dyn * arb / stuffable))
    nominal = arb + dyn_arb + 13
    data_bits = (1 + 4 + 8 * payload) + (dyn - dyn_arb) + 4 + crc + 1 + int(math.ceil(crc / 4))
    return nominal, data_bits


def frame_time(payload, bus, dyn=None):
    nb, db = canfd_bits(payload, dyn)
    return nb / CAN_BUSES[bus]["nominal"] + db / CAN_BUSES[bus]["data"]


def frame_times(payloads, bus, rng):
    """Vectorized durations with random dynamic stuffing ~ Binomial(stuffable bits, STUFF_P)."""
    out = np.empty(len(payloads))
    for P_ in np.unique(payloads):
        m = payloads == P_
        stuffable = 17 + 1 + 4 + 8 * int(P_)
        dyn = rng.binomial(stuffable, STUFF_P, m.sum())
        out[m] = [frame_time(int(P_), bus, int(d)) for d in dyn] if m.sum() < 64 else \
            _vec_ft(int(P_), bus, dyn)
    return out


def _vec_ft(P_, bus, dyn):
    arb = 17
    crc = 17 if P_ <= 16 else 21
    stuffable = arb + 1 + 4 + 8 * P_
    dyn = np.minimum(dyn, (stuffable - 1) // 4)
    dyn_arb = np.round(dyn * arb / stuffable)
    nom = arb + dyn_arb + 13
    dat = (1 + 4 + 8 * P_) + (dyn - dyn_arb) + 4 + crc + 1 + math.ceil(crc / 4)
    return nom / CAN_BUSES[bus]["nominal"] + dat / CAN_BUSES[bus]["data"]


# SOME/IP deployment of the zone services (Adaptive side, ara::com)
SOMEIP_HEADER = 16                                  # bytes (PRS_SOMEIPProtocol)
ETH_OVERHEAD = 8 + 14 + 4 + 12 + 20 + 8             # preamble, MAC hdr, FCS, IFG, IPv4, UDP
SERVICES = [
    # (service name, service id, provider, eventgroup id, I-PDUs mapped to events)
    ("PowertrainStatus", 0x1101, "ZC_FRONT", 1, ["ECM_EngSts", "ECM_EngThm", "PCU_MgSts", "PCU_MgThm",
                                                 "DCDC_Sts", "TMS_Sts"]),
    ("ChassisStatus", 0x1102, "ZC_FRONT", 1, ["ESC_VehDyn", "ESC_WhlSpd", "ESC_Brk", "EPS_Sts"]),
    ("EnergyStorageStatus", 0x1201, "ZC_REAR", 1, ["BMS_PackSts", "BMS_Thm"]),
    ("BodyChassisStatus", 0x1202, "ZC_REAR", 1, ["SCU_Travel", "SCU_BodyAcc", "TPMS_Press", "CCU_Sts"]),
    ("DriverInput", 0x1103, "ZC_FRONT", 1, []),     # pedal: zone-local I/O, 10 ms event
    ("EnergyManagement", 0x2001, "CVC", 1, []),      # CVC -> zones: torque / thermal / climate requests
]
