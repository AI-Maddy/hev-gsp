"""Frame-level simulation of the E/E layer.

* COM transmission per I-PDU: PERIODIC, or MIXED = periodic + TRIGGERED_ON_CHANGE
  with filter MASKED_NEW_DIFFERS_MASKED_OLD on the raw (CompuMethod-scaled)
  signal and ComMinimumDelayTime (AUTOSAR CP SWS COM).
* OS: frames are released by Com_MainFunctionTx; release jitter = COM task
  response time under the ECU's CPU load (fixed-priority preemptive OS).
* CAN FD: non-preemptive priority arbitration (lowest ID wins) with
  ISO 11898-1:2015 frame lengths and random bit stuffing. The CanIf transmit
  buffer holds one instance per PDU; a newer instance overwrites an unsent one
  (the lost instance appears to the E2E receiver as a counter jump).
* Zonal gateway: FIFO server (PduR routing + signal-to-service translation).
* Backbone: 100BASE-T1 utilization with background streams; SOME/IP events.
* CVC (Adaptive Platform): ara::com event dispatcher as FIFO server; PHM
  deadline supervision of the energy-management app.
* TCU (Adaptive Platform): telemetry uplink queue over a varying cellular link.

Graph-node outputs per 100 ms window:
  CP ECU  -> mean end-to-end latency sensor-sample -> CAN reception (ms)
  ZC_*    -> mean gateway sojourn time (ms)
  CVC     -> mean end-to-end latency sensor-sample -> ara::com consumer (ms)
  TCU     -> mean telemetry uplink latency (ms)
  CANFD_* -> bus utilization (%);  ETH_BB -> backbone utilization (%)
"""
import numpy as np
from numba import njit
from . import ee_arch as EA
from .faults import lvl

WIN = 0.1
# I-PDUs consumed by the energy-management Adaptive Application every 10 ms cycle (PHM-supervised)
SUPERVISED = {"ECM_EngSts", "PCU_MgSts", "ESC_VehDyn", "ESC_WhlSpd", "BMS_PackSts", "EPS_Sts"}


@njit(cache=True)
def _arbitrate(release, pdu, prio, dur, n_pdu):
    n = release.shape[0]
    start = np.full(n, -1.0)
    finish = np.full(n, -1.0)
    pend = np.full(n_pdu, -1, np.int64)
    npend = 0
    t = 0.0
    i = 0
    while True:
        if npend == 0:
            if i >= n:
                break
            if release[i] > t:
                t = release[i]
        while i < n and release[i] <= t:
            p = pdu[i]
            if pend[p] >= 0:
                finish[pend[p]] = -2.0          # overwritten in CanIf buffer (lost)
            else:
                npend += 1
            pend[p] = i
            i += 1
        best = -1
        bp = 1 << 30
        for p in range(n_pdu):
            if pend[p] >= 0 and prio[p] < bp:
                bp = prio[p]
                best = p
        k = pend[best]
        start[k] = t
        t += dur[k]
        finish[k] = t
        pend[best] = -1
        npend -= 1
    return start, finish


@njit(cache=True)
def _fifo(arrival, service):
    n = arrival.shape[0]
    dep = np.empty(n)
    t = 0.0
    for k in range(n):
        if arrival[k] > t:
            t = arrival[k]
        t += service[k]
        dep[k] = t
    return dep


def _ou(n, tau, sigma, rng):
    a = np.exp(-WIN / tau)
    e = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _win_mean(times, values, T, fill=np.nan):
    w = np.floor(times / WIN).astype(np.int64)
    ok = (w >= 0) & (w < T)
    s = np.bincount(w[ok], values[ok], minlength=T)
    c = np.bincount(w[ok], minlength=T)
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    # forward-fill empty windows (receiver keeps last value)
    idx = np.where(~np.isnan(m), np.arange(T), 0)
    np.maximum.accumulate(idx, out=idx)
    m = m[idx]
    m[np.isnan(m)] = fill if not np.isnan(fill) else np.nanmean(m) if np.any(~np.isnan(m)) else 0.0
    return m


def _win_count(times, T):
    w = np.floor(times / WIN).astype(np.int64)
    ok = (w >= 0) & (w < T)
    return np.bincount(w[ok], minlength=T)


def simulate_ee(X_phys, phys_names, t, pdus, faults=(), seed=0):
    rng = np.random.default_rng(seed + 101)
    T = len(t)
    t_end = T * WIN
    col = {n: i for i, n in enumerate(phys_names)}
    lv = lambda name: np.array([lvl(faults, name, tt) for tt in t])
    babble, zc_flood, bms_over = lv("ee_babbling_ESC"), lv("ee_zonal_overload"), lv("ee_bms_overrun")

    # ---- 1. nominal (sampling) times of every PDU instance
    frames = []     # per PDU: nominal times
    for p in pdus:
        P = p.period_ms / 1000
        off = rng.uniform(0, P)
        tn = np.arange(off, t_end, P)
        if p.mode == "MIXED" and p.signals:
            mt = EA.ECU_TIMING[p.sender][0] / 1000
            step = max(mt, p.mdt_ms / 1000)             # triggers land on Com_MainFunctionTx ticks >= MDT apart
            cnt = np.zeros(T)
            for s in p.signals:
                raw = np.floor((X_phys[:, col[s.name]] - s.offset) / s.factor).astype(np.int64)
                masked = raw >> s.mask_bits
                cnt += np.abs(np.diff(masked, prepend=masked[0]))
            ticks = int(round(WIN / step))
            cnt = np.minimum(cnt, ticks).astype(np.int64)
            trig = []
            for w in np.nonzero(cnt)[0]:
                k = rng.choice(ticks, cnt[w], replace=False)
                trig.append(w * WIN + (np.sort(k) + 1) * step)
            trig = np.concatenate(trig) if trig else np.zeros(0)
            tn = np.sort(np.concatenate([tn, trig]))
            # ComMinimumDelayTime across periodic and triggered sends
            if p.mdt_ms > 0 and len(tn) > 1:
                keep = np.ones(len(tn), bool)
                last = -1.0
                for q in range(len(tn)):
                    if tn[q] - last < p.mdt_ms / 1000 - 1e-9:
                        keep[q] = False
                    else:
                        last = tn[q]
                tn = tn[keep]
        frames.append(tn)
    # babbling-idiot frames (ESC, high priority, not a configured PDU)
    lam = 3500.0 * babble
    nb = rng.poisson(lam * WIN)
    bab = (np.repeat(np.arange(T), nb) + rng.uniform(0, 1, nb.sum())) * WIN
    bab_pdu = EA.IPdu("ESC_Babble", "ESC", "CANFD_F", 0x081, 0, "NONE", 0, 8, [], "", False)
    pdus_all = list(pdus) + [bab_pdu]
    frames.append(np.sort(bab))

    # ---- 2. ECU CPU load and COM task release jitter
    senders = sorted(set(p.sender for p in pdus_all))
    rate = {e: np.zeros(T) for e in senders}
    for p, tn in zip(pdus_all, frames):
        rate[p.sender] += _win_count(tn, T) / WIN
    u = {}
    for e in senders:
        mt, C, u0, c = EA.ECU_TIMING[e]
        ue = u0 + c * rate[e] + 0.015 * _ou(T, 20, 1, rng)
        if e == "BMS":
            ue = ue + 0.25 * bms_over
        if e == "ESC":
            ue = ue + 0.20 * babble
        u[e] = np.clip(ue, 0.05, 0.93)
    rel = []
    for p, tn in zip(pdus_all, frames):
        mt, C, *_ = EA.ECU_TIMING[p.sender]
        w = np.minimum((tn / WIN).astype(np.int64), T - 1)
        ue = u[p.sender][w]
        # response time: C + interference, mean C*u/(1-u) (gamma, shape 2)
        R = C + rng.gamma(2.0, 0.5 * C * ue / (1 - ue))
        rel.append(np.maximum.accumulate(tn + R / 1000))     # same task: FIFO order per PDU

    # ---- 3. CAN FD arbitration per bus
    out, aux = {}, {}
    arrivals = {z: [] for z in EA.ZONES}       # routed receptions at each zonal gateway
    e2e_err = {e: np.zeros(T) for e in EA.CP_ECUS}
    lat_ecu = {e: ([], []) for e in EA.CP_ECUS}
    for bus in EA.CAN_BUSES:
        ids = [k for k, p in enumerate(pdus_all) if p.bus == bus]
        tn = np.concatenate([frames[k] for k in ids])
        r = np.concatenate([rel[k] for k in ids])
        loc = np.concatenate([np.full(len(frames[k]), j) for j, k in enumerate(ids)])
        pay = np.array([pdus_all[k].payload for k in ids])[loc]
        dur = EA.frame_times(pay, bus, rng)            # ISO 11898-1 lengths, random stuffing
        order = np.argsort(r, kind="stable")
        tn, r, loc, dur = tn[order], r[order], loc[order], dur[order]
        prio = np.array([pdus_all[k].can_id for k in ids], np.int64)
        start, fin = _arbitrate(r, loc.astype(np.int64), prio, dur, len(ids))
        sent = fin > 0
        busy = np.bincount(np.minimum((start[sent] / WIN).astype(np.int64), T - 1),
                           dur[sent], minlength=T)
        out[bus] = 100 * busy / WIN
        for j, k in enumerate(ids):
            p = pdus_all[k]
            m = loc == j
            if p.sender in lat_ecu and p.name != "ESC_Babble":
                ms = m & sent
                lat_ecu[p.sender][0].append(fin[ms])
                lat_ecu[p.sender][1].append((fin[ms] - tn[ms]) * 1000)
            if p.e2e and p.sender in e2e_err:
                # E2E receiver check: lost instance -> counter jump (WRONGSEQUENCE with
                # MaxDeltaCounter = 1); reception gap > ComTimeout -> timeout
                lost = m & (fin == -2.0)
                e2e_err[p.sender] += _win_count(tn[lost], T)
                ft = np.sort(fin[m & sent])
                if len(ft) > 1:
                    gap = np.diff(ft) * 1000
                    e2e_err[p.sender] += _win_count(ft[1:][gap > p.timeout_ms], T)
            if p.routed:
                zc = EA.ZONE_OF[p.sender]
                ms = m & sent
                sup = p.name in SUPERVISED
                arrivals[zc].append((fin[ms], tn[ms], np.full(ms.sum(), len(p.signals)),
                                     np.full(ms.sum(), p.payload), np.full(ms.sum(), sup)))
    for e in EA.CP_ECUS:
        tt = np.concatenate(lat_ecu[e][0]); vv = np.concatenate(lat_ecu[e][1])
        out[e] = _win_mean(tt, vv, T)

    # ---- 4. zonal gateways (PduR routing + CAN->SOME/IP translation)
    events = []                                   # (time at ETH, sample time, payload)
    for zc in EA.ZONES:
        a = np.concatenate([x[0] for x in arrivals[zc]])
        tn = np.concatenate([x[1] for x in arrivals[zc]])
        ns = np.concatenate([x[2] for x in arrivals[zc]])
        pl = np.concatenate([x[3] for x in arrivals[zc]])
        sv = np.concatenate([x[4] for x in arrivals[zc]])
        kind = np.zeros(len(a), np.int8)
        if zc == "ZC_FRONT":                      # pedal: zone-local I/O sampled every 10 ms
            tp = np.arange(0.003, t_end, 0.010)
            a = np.concatenate([a, tp + 0.0002]); tn = np.concatenate([tn, tp])
            ns = np.concatenate([ns, np.ones(len(tp))]); pl = np.concatenate([pl, np.full(len(tp), 8)])
            kind = np.concatenate([kind, np.zeros(len(tp), np.int8)])
            sv = np.concatenate([sv, np.ones(len(tp), bool)])
            nf = rng.poisson(5000.0 * zc_flood * WIN)   # DoIP (ISO 13400) UDS request flood
            tf = (np.repeat(np.arange(T), nf) + rng.uniform(0, 1, nf.sum())) * WIN
            a = np.concatenate([a, tf]); tn = np.concatenate([tn, tf])
            ns = np.concatenate([ns, np.zeros(len(tf))]); pl = np.concatenate([pl, np.full(len(tf), 16)])
            kind = np.concatenate([kind, np.ones(len(tf), np.int8)])
            sv = np.concatenate([sv, np.zeros(len(tf), bool)])
        mt, C, u0, _ = EA.ECU_TIMING[zc]
        ubg = np.clip(u0 + 0.04 * _ou(T, 30, 1, rng), 0.05, 0.9)
        order = np.argsort(a, kind="stable")
        a, tn, ns, pl, kind, sv = a[order], tn[order], ns[order], pl[order], kind[order], sv[order]
        w = np.minimum((a / WIN).astype(np.int64), T - 1)
        svc = np.where(kind == 1, 0.08e-3, (40e-6 + 8e-6 * ns)) / (1 - ubg[w])
        dep = _fifo(a, svc)
        keep = kind == 0
        out[zc] = _win_mean(a[keep], (dep[keep] - a[keep]) * 1000, T)
        events.append((dep[keep], tn[keep], pl[keep], sv[keep]))

    # ---- 5. backbone (100BASE-T1) and SOME/IP events
    dep = np.concatenate([e[0] for e in events]); tn = np.concatenate([e[1] for e in events])
    pl = np.concatenate([e[2] for e in events])
    svc_sup = np.concatenate([e[3] for e in events])
    bits = (EA.ETH_OVERHEAD + EA.SOMEIP_HEADER + np.maximum(pl, 46 - 28 - 16)) * 8
    ev_load = np.bincount(np.minimum((dep / WIN).astype(np.int64), T - 1), bits, minlength=T) / WIN / EA.ETH["rate"]
    bg = np.clip(0.28 + 0.07 * _ou(T, 15, 1, rng), 0.05, 0.8)          # camera / infotainment streams
    flood_bits = 5000.0 * zc_flood * (EA.ETH_OVERHEAD + 8 + 16) * 8 / EA.ETH["rate"]
    rho = np.clip(bg + ev_load + flood_bits, 0, 0.95)
    out["ETH_BB"] = 100 * rho
    w = np.minimum((dep / WIN).astype(np.int64), T - 1)
    T_bg = 1542 * 8 / EA.ETH["rate"]                                     # max-size background frame
    eth_wait = rho[w] * T_bg / (2 * (1 - rho[w])) + bits / EA.ETH["rate"] + 5e-6   # M/D/1 + switch
    arr_cvc = dep + eth_wait

    # ---- 6. CVC (Adaptive Platform): ara::com dispatcher + PHM deadline supervision
    u_cvc = np.clip(0.40 + 0.08 * _ou(T, 25, 1, rng), 0.1, 0.9)        # ADAS / other AA load
    order = np.argsort(arr_cvc)
    arr_cvc, tn_c, sup_c = arr_cvc[order], tn[order], svc_sup[order]
    wc = np.minimum((arr_cvc / WIN).astype(np.int64), T - 1)
    done = _fifo(arr_cvc, 25e-6 / (1 - u_cvc[wc]))
    e2e_lat = (done - tn_c) * 1000
    out["CVC"] = _win_mean(done, e2e_lat, T)
    # PHM deadline supervision (CAPI phm DeadlineSupervision: minDeadline / maxDeadline) on the
    # energy-management inputs: sensor sample -> ara::com consumer, maxDeadline = 12 ms.
    PHM_MAX_DEADLINE_MS = 12.0
    aux["phm_deadline_fail"] = _win_count(done[sup_c & (e2e_lat > PHM_MAX_DEADLINE_MS)], T).astype(np.float32)

    # ---- 7. TCU (Adaptive Platform): telemetry uplink queue
    cell = np.clip(6e6 * np.exp(0.35 * _ou(T, 60, 1, rng)), 1e6, 20e6)  # bit/s
    tele = 0.05 * np.bincount(wc, bits, minlength=T) / WIN               # 5 % sampled + compressed
    tele = tele + 1.0e6 * zc_flood                                       # remote diag session traffic
    rho_u = np.clip(tele / cell, 0, 0.95)
    rtt = 35 + 10 * _ou(T, 90, 1, rng)
    out["TCU"] = rtt / 2 + 1000 * (8000 / cell) / (1 - rho_u)

    aux["e2e_errors"] = np.column_stack([e2e_err[e] for e in EA.CP_ECUS]).astype(np.float32)
    aux["cpu_load"] = np.column_stack([u[e] for e in EA.CP_ECUS + list(EA.ZONES)]).astype(np.float32)
    return out, aux
